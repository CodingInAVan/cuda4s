package flight4s.core.compiler

import munit.FunSuite

import flight4s.core.codegen.*
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.SourceSpan
import flight4s.core.unsafe.raw.RawCuda

class NvrtcCompilationInputSuite extends FunSuite:
  test("generated input retains module metadata and codegen provenance"):
    val signature = params(value[Int]("count"))
    val sourceMap = SourceMap(
      Vector(SourceMapEntry(1, SourceSpan("Kernel.scala", 1, 1, 1, 8)))
    )
    val options = CompilerOptions(
      additionalNvrtcOptions = Vector("--device-debug")
    )
    val requirements = KernelLaunchRequirements(
      Some(DynamicSharedMemoryRequirement(4, 4))
    )
    val generatedKernel = GeneratedKernel(
      name = "generatedKernel",
      signature = signature,
      cudaSource = "extern \"C\" __global__ void generatedKernel(int count) {}",
      sourceMap = sourceMap,
      compilerOptions = options,
      declarationLine = 1,
      launchRequirements = requirements
    )
    val module = GeneratedCudaModule(
      generatedKernel.cudaSource,
      sourceMap,
      options,
      Vector(generatedKernel)
    )

    val input = NvrtcCompilationInput.generated(module, codegenVersion = 19)

    assert(input.module eq module)
    assertEquals(input.provenance, NvrtcSourceProvenance.DslGenerated(19))
    assertEquals(input.source, module.cudaSource)
    assertEquals(input.sourceMap, sourceMap)
    assert(input.compilerOptions eq options)
    assertEquals(input.kernels.map(_.entryPoint), Vector("generatedKernel"))
    assert(input.kernels.head.signature eq signature)
    assertEquals(input.kernels.head.launchRequirements, requirements)

  test("raw input retains exact source with caller provenance and no Scala map"):
    val signature = params(value[Int]("count"), value[Float]("scale"))
    val source =
      "extern \"C\" __global__ void rawKernel(int count, float scale) {}\n"
    val options = CompilerOptions(
      additionalNvrtcOptions = Vector("--use_fast_math")
    )
    val requirements = KernelLaunchRequirements(
      Some(DynamicSharedMemoryRequirement(4, 4))
    )
    val definition = RawCuda.kernel(
      "rawKernel",
      signature,
      source,
      options,
      requirements
    )

    val input = NvrtcCompilationInput.raw(definition)

    assert(input.definition eq definition)
    assertEquals(input.provenance, NvrtcSourceProvenance.CallerProvidedRaw)
    assertEquals(input.source, source)
    assertEquals(input.sourceMap, SourceMap(Vector.empty))
    assert(input.compilerOptions eq options)
    assertEquals(input.kernels.map(_.entryPoint), Vector("rawKernel"))
    assert(input.kernels.head.signature eq signature)
    assertEquals(input.kernels.head.launchRequirements, requirements)

  test("compilation input factories reject invalid producer references"):
    val module = GeneratedCudaModule(
      "source",
      SourceMap(Vector.empty),
      CompilerOptions(),
      Vector.empty
    )

    intercept[IllegalArgumentException](
      NvrtcCompilationInput.generated(module, codegenVersion = 0)
    )
    intercept[IllegalArgumentException](
      NvrtcCompilationInput.generated(null)
    )
    intercept[IllegalArgumentException](
      NvrtcCompilationInput.raw(null)
    )
