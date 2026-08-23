package flight4s.core.unsafe.raw

import scala.compiletime.testing.typeCheckErrors

import munit.FunSuite

import flight4s.core.codegen.{
  CompilerOptions,
  DynamicSharedMemoryRequirement,
  KernelLaunchRequirements
}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.DeviceBuffer

class RawCudaSuite extends FunSuite:
  private val validSource =
    """extern "C" __global__
      |void saxpy(const float* x, float* y, float alpha) {
      |  int i = blockIdx.x * blockDim.x + threadIdx.x;
      |  y[i] = alpha * x[i] + y[i];
      |}
      |""".stripMargin

  test("raw CUDA definitions retain typed metadata and exact caller source"):
    val signature = params(
      input[Float]("x"),
      output[Float]("y"),
      value[Float]("alpha")
    )
    val options = CompilerOptions(
      additionalNvrtcOptions = Vector("--use_fast_math")
    )
    val requirements = KernelLaunchRequirements(
      dynamicSharedMemory = Some(DynamicSharedMemoryRequirement(4, 4))
    )
    val definition: RawCudaKernel[
      (DeviceBuffer[Float], DeviceBuffer[Float], Float)
    ] = RawCuda.kernel(
      entryPoint = "saxpy",
      signature = signature,
      source = validSource,
      compilerOptions = options,
      launchRequirements = requirements
    )

    assertEquals(definition.entryPoint, "saxpy")
    assert(definition.signature eq signature)
    assertEquals(definition.parameters.map(_.name), Vector("x", "y", "alpha"))
    assertEquals(definition.abiDescriptors, signature.abiDescriptors)
    assertEquals(definition.source, validSource)
    assert(definition.compilerOptions eq options)
    assert(definition.launchRequirements eq requirements)
    assertEquals(definition.toString, "RawCudaKernel(saxpy)")

  test("raw CUDA invocation reuses typed signature ABI packing"):
    val signature = params(value[Int]("count"), value[Float]("scale"))
    val definition = RawCuda.kernel(
      "scaleValues",
      signature,
      "extern \"C\" __global__ void scaleValues(int count, float scale) {}",
      CompilerOptions(),
      KernelLaunchRequirements()
    )
    val invocation = definition.bind((128, 0.5f))

    assert(invocation.kernel eq definition)
    assertEquals(invocation.arguments, (128, 0.5f))
    assertEquals(
      invocation.packedArguments.descriptors,
      signature.abiDescriptors
    )
    assertEquals(
      invocation.packedArguments.arguments.map(_.bytes.size),
      Vector(4, 4)
    )

  test("raw CUDA invocation rejects incorrect Scala argument types"):
    val errors = typeCheckErrors(
      """
        import flight4s.core.codegen.*
        import flight4s.core.dsl.CudaDsl.*
        import flight4s.core.unsafe.raw.RawCuda

        val definition = RawCuda.kernel(
          "scaleValues",
          params(value[Int]("count"), value[Float]("scale")),
          "extern \"C\" __global__ void scaleValues(int count, float scale) {}",
          CompilerOptions(),
          KernelLaunchRequirements()
        )
        val invalid = definition.bind((128, true))
      """
    )

    assert(errors.nonEmpty)

  test("raw CUDA definitions reject invalid boundary metadata"):
    def expect(
        expected: RawCudaDefinitionErrorCode
    )(definition: => RawCudaKernel[?]): Unit =
      val error = intercept[RawCudaDefinitionError](definition)
      assertEquals(error.code, expected)

    expect(RawCudaDefinitionErrorCode.InvalidEntryPoint) {
      RawCuda.kernel(
        "invalid entry",
        params(),
        validSource,
        CompilerOptions(),
        KernelLaunchRequirements()
      )
    }
    expect(RawCudaDefinitionErrorCode.EmptySource) {
      RawCuda.kernel(
        "emptySource",
        params(),
        " \r\n\t",
        CompilerOptions(),
        KernelLaunchRequirements()
      )
    }
    expect(RawCudaDefinitionErrorCode.SourceContainsNullCharacter) {
      RawCuda.kernel(
        "nullSource",
        params(),
        validSource + '\u0000',
        CompilerOptions(),
        KernelLaunchRequirements()
      )
    }
    expect(RawCudaDefinitionErrorCode.InvalidParameterName) {
      RawCuda.kernel(
        "invalidParameter",
        params(value[Int]("not valid")),
        validSource,
        CompilerOptions(),
        KernelLaunchRequirements()
      )
    }
    expect(RawCudaDefinitionErrorCode.DuplicateParameterName) {
      RawCuda.kernel(
        "duplicateParameter",
        params(value[Int]("size"), value[Float]("size")),
        validSource,
        CompilerOptions(),
        KernelLaunchRequirements()
      )
    }

  test("raw CUDA remains outside typed IR and code generation"):
    val moduleErrors = typeCheckErrors(
      """
        import flight4s.core.codegen.*
        import flight4s.core.dsl.CudaDsl.*
        import flight4s.core.unsafe.raw.RawCuda

        val rawKernel = RawCuda.kernel(
          "rawKernel",
          params(),
          "extern \"C\" __global__ void rawKernel() {}",
          CompilerOptions(),
          KernelLaunchRequirements()
        )
        val invalid = module(kernels = Vector(rawKernel))
      """
    )
    val codegenErrors = typeCheckErrors(
      """
        import flight4s.core.codegen.*
        import flight4s.core.dsl.CudaDsl.*
        import flight4s.core.unsafe.raw.RawCuda

        val rawKernel = RawCuda.kernel(
          "rawKernel",
          params(),
          "extern \"C\" __global__ void rawKernel() {}",
          CompilerOptions(),
          KernelLaunchRequirements()
        )
        val invalid = CudaCodegen.generate(rawKernel)
      """
    )
    val normalizerErrors = typeCheckErrors(
      """
        import flight4s.core.codegen.*
        import flight4s.core.dsl.CudaDsl.*
        import flight4s.core.ir.IrNormalizer
        import flight4s.core.unsafe.raw.RawCuda

        val rawKernel = RawCuda.kernel(
          "rawKernel",
          params(),
          "extern \"C\" __global__ void rawKernel() {}",
          CompilerOptions(),
          KernelLaunchRequirements()
        )
        val invalid = IrNormalizer.kernel(rawKernel)
      """
    )

    assert(moduleErrors.nonEmpty)
    assert(codegenErrors.nonEmpty)
    assert(normalizerErrors.nonEmpty)

  test("raw definition validation does not pretend to parse CUDA C++"):
    val callerSource = "reviewed by NVRTC in the compilation slice"
    val definition = RawCuda.kernel(
      "declaredEntryPoint",
      params(),
      callerSource,
      CompilerOptions(),
      KernelLaunchRequirements()
    )

    assertEquals(definition.source, callerSource)
