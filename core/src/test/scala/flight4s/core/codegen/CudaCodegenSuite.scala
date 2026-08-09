package flight4s.core.codegen

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.Comparator

import munit.FunSuite

import flight4s.core.codegen.CodegenError.ValidationFailed
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.types.*

class CudaCodegenSuite extends FunSuite:
  test("vectorAdd generates deterministic inspectable CUDA C++"):
    val left = input[Float]("left")
    val right = input[Float]("right")
    val outputBuffer = output[Float]("output")
    val elementCount = value[Int]("elementCount")
    val definition = kernel(
      "vectorAdd",
      params(left, right, outputBuffer, elementCount)
    ) { bindings =>
      val boundLeft = bindings.head
      val boundRight = bindings.tail.head
      val boundOutput = bindings.tail.tail.head
      val boundElementCount = bindings.tail.tail.tail.head
      val index = local(
        "index",
        blockIdx.x * blockDim.x + threadIdx.x
      )

      when(index.read < boundElementCount) {
        boundOutput(index.read) :=
          boundLeft(index.read).read + boundRight(index.read).read
      }
    }

    val generated = generatedKernel(definition)

    assertEquals(
      generated.cudaSource,
      """extern "C" __global__ void vectorAdd(const float* left, const float* right, float* output, int elementCount) {
        |  int index = ((blockIdx.x * blockDim.x) + threadIdx.x);
        |  if ((index < elementCount)) {
        |    output[index] = (left[index] + right[index]);
        |  }
        |}
        |""".stripMargin.replace("\r\n", "\n")
    )
    assertEquals(generated.name, "vectorAdd")
    assertEquals(generated.signature, definition.signature)
    assertEquals(generated.declarationLine, 1)
    assertEquals(
      generated.compilerOptions.nvrtcOptions,
      Vector("--std=c++20")
    )
    assertEquals(
      generated.launchRequirements,
      KernelLaunchRequirements()
    )

  test("module generation preserves constant and shared-memory ownership"):
    val coefficients = constantArray[Float]("coefficients", 64)
    val outputBuffer = output[Float]("output")
    val definition = kernel("memorySpaces", params(outputBuffer)) { bindings =>
      val output = bindings.head
      val tile = sharedArray2D[Float](
        "tile",
        rows = 32,
        columns = 32,
        rowStride = 33
      )
      val cube = sharedArray3D[Float](
        "cube",
        depth = 8,
        rows = 32,
        columns = 32,
        rowStride = 33
      )
      val dynamicTile = dynamicSharedArray[Float]("dynamicTile")
      val scratch = localArray[Float]("scratch", 2)

      tile(threadIdx.y, threadIdx.x) :=
        coefficients(threadIdx.x).read
      cube(blockIdx.z, threadIdx.y, threadIdx.x) := literal(1.0f)
      scratch(literal(0)) := tile(threadIdx.y, threadIdx.x).read
      dynamicTile(threadIdx.x) := scratch(literal(0)).read
      barrier()
      output(threadIdx.x) := dynamicTile(threadIdx.x).read
    }

    val generated = generatedModule(
      module(
        constants = Vector(coefficients),
        kernels = Vector(definition)
      )
    )

    assertEquals(
      generated.cudaSource,
      """extern "C" __constant__ float coefficients[64];
        |
        |extern "C" __global__ void memorySpaces(float* output) {
        |  __shared__ float tile[32][33];
        |  __shared__ float cube[8][32][33];
        |  extern __shared__ __align__(4) float dynamicTile[];
        |
        |  float scratch[2];
        |  tile[threadIdx.y][threadIdx.x] = coefficients[threadIdx.x];
        |  cube[blockIdx.z][threadIdx.y][threadIdx.x] = 0x1.0p0f;
        |  scratch[0] = tile[threadIdx.y][threadIdx.x];
        |  dynamicTile[threadIdx.x] = scratch[0];
        |  __syncthreads();
        |  output[threadIdx.x] = dynamicTile[threadIdx.x];
        |}
        |""".stripMargin.replace("\r\n", "\n")
    )
    assertEquals(generated.kernels.map(_.name), Vector("memorySpaces"))
    assertEquals(generated.kernels.head.declarationLine, 3)
    assertEquals(
      generated.kernels.head.launchRequirements,
      KernelLaunchRequirements(
        Some(
          DynamicSharedMemoryRequirement(
            elementSizeBytes = 4,
            elementAlignmentBytes = 4
          )
        )
      )
    )

  test("dynamic shared-memory tree reduction emits staged barriers"):
    val inputBuffer = input[Float]("input")
    val outputBuffer = output[Float]("output")
    val definition = kernel(
      "blockReduceSum",
      params(inputBuffer, outputBuffer)
    ) { bindings =>
      val input = bindings.head
      val output = bindings.tail.head
      val scratch = dynamicSharedArray[Float]("scratch")

      scratch(threadIdx.x) := input(threadIdx.x).read
      barrier()
      Vector(128, 64, 32, 16, 8, 4, 2, 1).foreach { stride =>
        when(threadIdx.x < literal(stride)) {
          scratch(threadIdx.x) :=
            scratch(threadIdx.x).read +
              scratch(threadIdx.x + literal(stride)).read
        }
        barrier()
      }
      when(threadIdx.x === literal(0)) {
        output(literal(0)) := scratch(literal(0)).read
      }
    }

    val generated = generatedKernel(definition)

    assertEquals(
      generated.cudaSource,
      """extern "C" __global__ void blockReduceSum(const float* input, float* output) {
        |  extern __shared__ __align__(4) float scratch[];
        |
        |  scratch[threadIdx.x] = input[threadIdx.x];
        |  __syncthreads();
        |  if ((threadIdx.x < 128)) {
        |    scratch[threadIdx.x] = (scratch[threadIdx.x] + scratch[(threadIdx.x + 128)]);
        |  }
        |  __syncthreads();
        |  if ((threadIdx.x < 64)) {
        |    scratch[threadIdx.x] = (scratch[threadIdx.x] + scratch[(threadIdx.x + 64)]);
        |  }
        |  __syncthreads();
        |  if ((threadIdx.x < 32)) {
        |    scratch[threadIdx.x] = (scratch[threadIdx.x] + scratch[(threadIdx.x + 32)]);
        |  }
        |  __syncthreads();
        |  if ((threadIdx.x < 16)) {
        |    scratch[threadIdx.x] = (scratch[threadIdx.x] + scratch[(threadIdx.x + 16)]);
        |  }
        |  __syncthreads();
        |  if ((threadIdx.x < 8)) {
        |    scratch[threadIdx.x] = (scratch[threadIdx.x] + scratch[(threadIdx.x + 8)]);
        |  }
        |  __syncthreads();
        |  if ((threadIdx.x < 4)) {
        |    scratch[threadIdx.x] = (scratch[threadIdx.x] + scratch[(threadIdx.x + 4)]);
        |  }
        |  __syncthreads();
        |  if ((threadIdx.x < 2)) {
        |    scratch[threadIdx.x] = (scratch[threadIdx.x] + scratch[(threadIdx.x + 2)]);
        |  }
        |  __syncthreads();
        |  if ((threadIdx.x < 1)) {
        |    scratch[threadIdx.x] = (scratch[threadIdx.x] + scratch[(threadIdx.x + 1)]);
        |  }
        |  __syncthreads();
        |  if ((threadIdx.x == 0)) {
        |    output[0] = scratch[0];
        |  }
        |}
        |""".stripMargin.replace("\r\n", "\n")
    )
    assertEquals(
      generated.launchRequirements,
      KernelLaunchRequirements(
        Some(
          DynamicSharedMemoryRequirement(
            elementSizeBytes = 4,
            elementAlignmentBytes = 4
          )
        )
      )
    )

  test("low-precision expressions emit required headers and CUDA conversions"):
    val halfInput = input[Float16]("halfInput")
    val bfloatInput = input[BFloat16]("bfloatInput")
    val fp8Input = input[Float8E4M3]("fp8Input")
    val outputBuffer = output[Float]("output")
    val definition = kernel(
      "lowPrecision",
      params(halfInput, bfloatInput, fp8Input, outputBuffer)
    ) { bindings =>
      val half = bindings.head
      val bfloat = bindings.tail.head
      val fp8 = bindings.tail.tail.head
      val output = bindings.tail.tail.tail.head
      local(
        "convertedHalf",
        convert.f32ToF16(
          literal(1.0f),
          RoundingMode.TowardZero
        )
      )
      local(
        "convertedBfloat",
        convert.f32ToBF16(
          literal(1.0f),
          RoundingMode.TowardPositive
        )
      )
      local(
        "convertedFp8",
        convert.f32ToFP8E4M3(
          literal(1.0f),
          SaturationMode.NoSaturation
        )
      )
      output(literal(0)) :=
        half(literal(0)).read.toAccumulator +
          bfloat(literal(0)).read.toAccumulator +
          fp8(literal(0)).read.toAccumulator
    }

    val source = generatedKernel(definition).cudaSource

    assert(source.startsWith(
      """#include <cuda_bf16.h>
        |#include <cuda_fp16.h>
        |#include <cuda_fp8.h>
        |""".stripMargin.replace("\r\n", "\n")
    ))
    assert(source.contains("__float2half_rz(0x1.0p0f)"))
    assert(source.contains("__float2bfloat16_ru(0x1.0p0f)"))
    assert(source.contains("__NV_NOSAT, __NV_E4M3"))
    assert(source.contains("__half2float(halfInput[0])"))
    assert(source.contains("__bfloat162float(bfloatInput[0])"))
    assert(source.contains("static_cast<float>(fp8Input[0])"))

  test("reduceSum lowers to a collision-free device-side expression"):
    val source = input[Float16]("source")
    val outputBuffer = output[Float]("output")
    val definition = kernel(
      "sumValues",
      params(source, outputBuffer)
    ) { bindings =>
      val boundSource = bindings.head
      val boundOutput = bindings.tail.head
      local("flight4s_accumulator_0", literal(0.0f))
      val sum = reduceSum(
        "i",
        literal(0),
        literal(32),
        literal(0.0f)
      ) { index =>
        boundSource(index).read
      }
      boundOutput(literal(0)) := sum
    }

    val sourceCode = generatedKernel(definition).cudaSource

    assert(sourceCode.contains("float flight4s_accumulator_0 = 0x0.0p0f;"))
    assert(sourceCode.contains(
      "float flight4s_accumulator_1 = 0x0.0p0f;"
    ))
    assert(sourceCode.contains(
      "flight4s_accumulator_1 += __half2float(source[i]);"
    ))

  test("generated temporary names cannot shadow module constants"):
    val accumulatorTable =
      constantArray[Float]("flight4s_accumulator_0", 4)
    val valueTable = constantArray[Float]("flight4s_value_0", 1)
    val outputBuffer = output[Float]("output")
    val definition = kernel(
      "internalNames",
      params(outputBuffer)
    ) { bindings =>
      val output = bindings.head
      local(
        "converted",
        convert.f32ToFP8E4M3(
          valueTable(literal(0)).read
        )
      )
      val sum = reduceSum(
        "i",
        literal(0),
        literal(4),
        literal(0.0f)
      ) { index =>
        accumulatorTable(index).read
      }
      output(literal(0)) := sum
    }

    val source = generatedModule(
      module(
        constants = Vector(accumulatorTable, valueTable),
        kernels = Vector(definition)
      )
    ).cudaSource

    assert(source.contains("__nv_fp8_e4m3 flight4s_value_1;"))
    assert(source.contains("flight4s_value_0[0]"))
    assert(source.contains("float flight4s_accumulator_2 = 0x0.0p0f;"))
    assert(source.contains(
      "flight4s_accumulator_2 += flight4s_accumulator_0[i];"
    ))

  test("source maps retain known source positions by generated line"):
    val constantSpan = SourceSpan("Kernel.scala", 3, 5, 3, 40)
    val storeSpan = SourceSpan("Kernel.scala", 9, 7, 9, 31)
    val table = ConstantArray("table", F32, 4, constantSpan)
    val outputBuffer = output[Float]("output")
    val definition = KernelIR(
      "mapped",
      params(outputBuffer),
      Block(
        Vector(
          Store(
            BufferElement[Float, ReadWrite](
              "output",
              literal(0),
              F32
            ),
            Load(ConstantElement("table", literal(0), F32)),
            storeSpan
          )
        )
      )
    )

    val generated = generatedModule(
      CudaModuleIR(Vector(table), Vector(definition))
    )

    assertEquals(
      generated.sourceMap.entries,
      Vector(
        SourceMapEntry(1, constantSpan),
        SourceMapEntry(4, storeSpan)
      )
    )
    assertEquals(
      generated.sourceMap.closestAtOrBefore(5),
      Some(SourceMapEntry(4, storeSpan))
    )

  test("code generation normalizes literals without changing statement source maps"):
    val storeSpan = SourceSpan("Normalized.scala", 7, 5, 7, 41)
    val index = Binary(
      BinaryOperator.Add,
      Literal(1, I32),
      Literal(2, I32),
      I32
    )
    val value = Binary(
      BinaryOperator.Multiply,
      Literal(3, I32),
      Literal(4, I32),
      I32
    )
    val outputBuffer = output[Int]("output")
    val definition = KernelIR(
      "normalizedLiterals",
      params(outputBuffer),
      Block(
        Vector(
          Store(
            BufferElement[Int, ReadWrite]("output", index, I32),
            value,
            storeSpan
          )
        )
      )
    )

    val generated = generatedModule(CudaModuleIR(Vector.empty, Vector(definition)))

    assertEquals(
      generated.cudaSource,
      """extern "C" __global__ void normalizedLiterals(int* output) {
        |  output[3] = 12;
        |}
        |""".stripMargin.replace("\r\n", "\n")
    )
    assertEquals(
      generated.sourceMap.entries,
      Vector(SourceMapEntry(2, storeSpan))
    )
    val originalStore = definition.body.statements.head.asInstanceOf[Store[Int, Global]]
    assertEquals(originalStore.to.asInstanceOf[BufferElement[Int, ReadWrite]].index, index)
    assertEquals(originalStore.value, value)

  test("source generation is gated by module validation"):
    val invalid = KernelIR(
      "invalid-name",
      params(),
      Block(Vector.empty)
    )

    CudaCodegen.generateModule(
      CudaModuleIR(Vector.empty, Vector(invalid))
    ) match
      case Left(ValidationFailed(errors)) =>
        assertEquals(
          errors.map(_.code),
          Vector(ValidationCode.InvalidKernelName)
        )
      case other =>
        fail(s"expected validation failure, got $other")

  test("compiler options have one authoritative CUDA C++ standard"):
    val options = CompilerOptions(
      additionalNvrtcOptions = Vector("--use_fast_math")
    )

    assertEquals(
      options.nvrtcOptions,
      Vector("--std=c++20", "--use_fast_math")
    )
    intercept[IllegalArgumentException](
      CompilerOptions(
        additionalNvrtcOptions = Vector("--std=c++17")
      )
    )
    intercept[IllegalArgumentException](
      CompilerOptions(
        additionalNvrtcOptions = Vector("-std=c++17")
      )
    )
    intercept[IllegalArgumentException](
      CompilerOptions(
        additionalNvrtcOptions =
          Vector("--gpu-architecture=compute_80")
      )
    )
    intercept[IllegalArgumentException](
      CompilerOptions(
        additionalNvrtcOptions = Vector("-arch=compute_80")
      )
    )
    intercept[IllegalArgumentException](
      CompilerOptions(additionalNvrtcOptions = Vector(""))
    )
    intercept[IllegalArgumentException](
      CompilerOptions(
        additionalNvrtcOptions = Vector("--define=bad\u0000value")
      )
    )

  test("representative generated source compiles to PTX when nvcc is available"):
    val nvccPath = findNvcc()
    assume(nvccPath.nonEmpty, "nvcc is not available")
    val nvcc = nvccPath.get
    val coefficients = constantArray[Float]("coefficients", 64)
    val halfInput = input[Float16]("halfInput")
    val outputBuffer = output[Float]("output")
    val definition = kernel(
      "compileGeneratedSource",
      params(halfInput, outputBuffer)
    ) { bindings =>
      val input = bindings.head
      val output = bindings.tail.head
      val tile = sharedArray2D[Float](
        "tile",
        rows = 32,
        columns = 32,
        rowStride = 33
      )
      val dynamicTile = dynamicSharedArray[Float]("dynamicTile")
      val index = threadIdx.x
      local(
        "convertedHalf",
        convert.f32ToF16(literal(1.0f))
      )
      local(
        "convertedBfloat",
        convert.f32ToBF16(literal(1.0f))
      )
      local(
        "convertedFp8",
        convert.f32ToFP8E4M3(literal(1.0f))
      )
      local(
        "halfBits",
        literal(Float16.fromBits(0x3c00.toShort))
      )
      local(
        "bfloatBits",
        literal(BFloat16.fromBits(0x3f80.toShort))
      )
      local(
        "fp8E4M3Bits",
        literal(Float8E4M3.fromBits(0x38.toByte))
      )
      local(
        "fp8E5M2Bits",
        literal(Float8E5M2.fromBits(0x3c.toByte))
      )
      val sum = reduceSum(
        "i",
        literal(0),
        literal(4),
        literal(0.0f)
      ) { reductionIndex =>
        input(reductionIndex).read
      }

      tile(threadIdx.y, index) := coefficients(index).read
      dynamicTile(index) := sum
      barrier()
      output(index) := dynamicTile(index).read
    }
    val generated = generatedModule(
      module(
        constants = Vector(coefficients),
        kernels = Vector(definition)
      )
    )
    val directory = Files.createTempDirectory("flight4s-codegen-")
    val source = directory.resolve("generated.cu")
    val ptx = directory.resolve("generated.ptx")

    try
      Files.writeString(
        source,
        generated.cudaSource,
        StandardCharsets.UTF_8
      )
      val process = ProcessBuilder(
        nvcc,
        "--std=c++20",
        "--ptx",
        source.toString,
        "-o",
        ptx.toString
      ).redirectErrorStream(true).start()
      val output = new String(
        process.getInputStream.readAllBytes(),
        StandardCharsets.UTF_8
      )
      val exitCode = process.waitFor()

      assertEquals(exitCode, 0, output)
      assert(Files.size(ptx) > 0L)
    finally
      deleteRecursively(directory)

  private def generatedKernel[Args <: Tuple](
      kernel: Kernel[Args]
  ): GeneratedKernel[Args] =
    CudaCodegen.generate(kernel) match
      case Right(generated) => generated
      case Left(error) => fail(error.message)

  private def generatedModule(
      module: CudaModuleIR
  ): GeneratedCudaModule =
    CudaCodegen.generateModule(module) match
      case Right(generated) => generated
      case Left(error) => fail(error.message)

  private def findNvcc(): Option[String] =
    val executable =
      if System.getProperty("os.name").startsWith("Windows") then "nvcc.exe"
      else "nvcc"
    val cudaHomeCandidates = Vector(
      Option(System.getenv("CUDA_PATH")),
      Option(System.getenv("CUDA_HOME"))
    ).flatten.map(home => Path.of(home, "bin", executable).toString)
    (cudaHomeCandidates :+ executable).find { candidate =>
      try
        val process = ProcessBuilder(candidate, "--version")
          .redirectErrorStream(true)
          .start()
        process.getInputStream.readAllBytes()
        process.waitFor() == 0
      catch
        case _: Exception => false
    }

  private def deleteRecursively(root: Path): Unit =
    if Files.exists(root) then
      val paths = Files.walk(root)
      try
        paths
          .sorted(Comparator.reverseOrder())
          .forEach(path => Files.deleteIfExists(path))
      finally paths.close()
