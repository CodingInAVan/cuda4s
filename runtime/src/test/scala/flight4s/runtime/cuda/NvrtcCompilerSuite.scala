package flight4s.runtime.cuda

import java.nio.charset.StandardCharsets

import munit.FunSuite

import flight4s.core.codegen.*
import flight4s.core.compiler.*
import flight4s.core.dsl.CudaDsl.*

class NvrtcCompilerSuite extends FunSuite:
  private val nativeLibraryConfigured =
    sys.props.contains("flight4s.cuda.native.path")

  test("generated CUDA compiles to an inspectable PTX artifact"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val inputBuffer = input[Float]("input")
    val outputBuffer = output[Float]("output")
    val definition = kernel(
      "copyValues",
      params(inputBuffer, outputBuffer)
    ) { bindings =>
      val input = bindings.head
      val output = bindings.tail.head
      output(threadIdx.x) := input(threadIdx.x).read
    }
    val generated = CudaCodegen.generate(definition) match
      case Right(value) =>
        GeneratedCudaModule(
          value.cudaSource,
          value.sourceMap,
          value.compilerOptions,
          Vector(value)
        )
      case Left(error) =>
        fail(error.message)

    NvrtcCompiler.compile(
      generated,
      ComputeCapability(8, 0),
      "copy_values.cu"
    ) match
      case Right(artifact) =>
        val ptxText = String(
          artifact.ptx.toArray,
          StandardCharsets.UTF_8
        )

        assert(artifact.ptx.nonEmpty)
        assert(ptxText.contains(".entry copyValues"))
        assert(artifact.nvrtcVersion.major >= 12)
        assertEquals(artifact.target, ComputeCapability(8, 0))
        assertEquals(artifact.programName, "copy_values.cu")
        assertEquals(
          artifact.compilerOptions.values,
          Vector(
            "--std=c++20",
            "--gpu-architecture=compute_80"
          )
        )
        assertEquals(artifact.generated, generated)
      case Left(failure) =>
        fail(failure.message + "\n" + failure.compileLog)

  test("NVRTC compilation failures retain source, options, and logs"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val generated = GeneratedCudaModule(
      cudaSource =
        """extern "C" __global__ void broken( {
          |}
          |""".stripMargin.replace("\r\n", "\n"),
      sourceMap = SourceMap(Vector.empty),
      compilerOptions = CompilerOptions(),
      kernels = Vector.empty
    )

    NvrtcCompiler.compile(
      generated,
      ComputeCapability(8, 0),
      "broken_kernel.cu"
    ) match
      case Left(failure) =>
        assertEquals(
          failure.resultName,
          "NVRTC_ERROR_COMPILATION"
        )
        assert(failure.resultCode != 0)
        assert(failure.compileLog.nonEmpty)
        assert(failure.compileLog.contains("broken_kernel.cu"))
        assertEquals(failure.generated, generated)
        assertEquals(
          failure.compilerOptions.values,
          Vector(
            "--std=c++20",
            "--gpu-architecture=compute_80"
          )
        )
      case Right(_) =>
        fail("invalid CUDA source unexpectedly compiled")

  test("compile request validates source and logical program name before JNI"):
    val empty = GeneratedCudaModule(
      cudaSource = "",
      sourceMap = SourceMap(Vector.empty),
      compilerOptions = CompilerOptions(),
      kernels = Vector.empty
    )
    val nonEmpty = empty.copy(cudaSource = " ")
    val target = ComputeCapability(8, 0)

    intercept[IllegalArgumentException](
      NvrtcCompiler.compile(empty, target)
    )
    intercept[IllegalArgumentException](
      NvrtcCompiler.compile(nonEmpty, target, "")
    )
    intercept[IllegalArgumentException](
      NvrtcCompiler.compile(nonEmpty, target, "bad\u0000name.cu")
    )
