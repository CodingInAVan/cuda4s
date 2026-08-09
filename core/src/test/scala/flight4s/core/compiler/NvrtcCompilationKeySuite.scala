package flight4s.core.compiler

import munit.FunSuite

import flight4s.core.codegen.*
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.SourceSpan

class NvrtcCompilationKeySuite extends FunSuite:
  private val target = ComputeCapability(8, 0)
  private val version = NvrtcVersion(13, 0)
  private val programName = "copy_values.cu"

  test("identical compilation inputs produce one stable SHA-256 key"):
    val generated = copyModule()

    val first = derive(generated)
    val second = derive(generated)

    assertEquals(first, second)
    assertEquals(first.hex.length, 64)
    assert(first.hex.forall { character =>
      character >= '0' && character <= '9' ||
      character >= 'a' && character <= 'f'
    })
    assertEquals(first.toString, first.hex)
    assertEquals(
      first.hex,
      "da97204e544e9a2d724a7449e988904fc454e31418a2924072a1eb5a9acc75c3"
    )

  test("every compiler-relevant input invalidates the key"):
    val generated = copyModule()
    val alternateMetadata = scalarModule()
    val baseKey = derive(generated)
    val changedOptions = generated.copy(
      compilerOptions = CompilerOptions(
        additionalNvrtcOptions = Vector("--use_fast_math")
      )
    )
    val changedMetadata = generated.copy(
      kernels = alternateMetadata.kernels
    )
    val changedLaunchRequirements = generated.copy(
      kernels = generated.kernels.map { kernel =>
        withLaunchRequirements(
          kernel,
          KernelLaunchRequirements(
            dynamicSharedMemory = Some(
              DynamicSharedMemoryRequirement(4, 4)
            )
          )
        )
      }
    )

    val variants = Vector(
      "CUDA source" -> derive(
        generated.copy(cudaSource = generated.cudaSource + "// changed\n")
      ),
      "compiler options" -> derive(changedOptions),
      "compute capability" -> derive(
        generated,
        target = ComputeCapability(9, 0)
      ),
      "NVRTC version" -> derive(
        generated,
        nvrtcVersion = NvrtcVersion(13, 1)
      ),
      "program name" -> derive(
        generated,
        programName = "renamed.cu"
      ),
      "codegen version" -> derive(
        generated,
        codegenVersion = CudaCodegen.ArtifactVersion + 1
      ),
      "kernel ABI metadata" -> derive(changedMetadata),
      "kernel launch requirements" -> derive(changedLaunchRequirements)
    )

    variants.foreach { case (label, variantKey) =>
      assert(baseKey != variantKey, s"$label did not invalidate the key")
    }

  test("canonical option encoding preserves element boundaries and order"):
    val generated = copyModule()
    val splitAfter = generated.copy(
      compilerOptions = CompilerOptions(
        additionalNvrtcOptions = Vector("ab", "c")
      )
    )
    val splitBefore = generated.copy(
      compilerOptions = CompilerOptions(
        additionalNvrtcOptions = Vector("a", "bc")
      )
    )
    val reordered = generated.copy(
      compilerOptions = CompilerOptions(
        additionalNvrtcOptions = Vector("c", "ab")
      )
    )

    assertNotEquals(derive(splitAfter), derive(splitBefore))
    assertNotEquals(derive(splitAfter), derive(reordered))

  test("source maps and generated line metadata do not affect PTX identity"):
    val generated = copyModule()
    val replacementMap = SourceMap(
      Vector(
        SourceMapEntry(
          generatedLine = 99,
          sourceSpan = SourceSpan("Other.scala", 80, 7, 80, 42)
        )
      )
    )
    val remapped = generated.copy(
      sourceMap = replacementMap,
      kernels = generated.kernels.map { kernel =>
        withSourceMetadata(kernel, replacementMap)
      }
    )

    assertEquals(derive(generated), derive(remapped))

  test("invalid key inputs are rejected before hashing"):
    val generated = copyModule()

    intercept[IllegalArgumentException](
      derive(generated.copy(cudaSource = ""))
    )
    intercept[IllegalArgumentException](
      derive(generated, programName = "")
    )
    intercept[IllegalArgumentException](
      derive(generated, programName = "bad\u0000name.cu")
    )
    intercept[IllegalArgumentException](
      derive(generated, codegenVersion = 0)
    )

  private def derive(
      generated: GeneratedCudaModule,
      target: ComputeCapability = target,
      nvrtcVersion: NvrtcVersion = version,
      programName: String = programName,
      codegenVersion: Int = CudaCodegen.ArtifactVersion
  ): NvrtcCompilationKey =
    NvrtcCompilationKey.derive(
      generated,
      target,
      nvrtcVersion,
      programName,
      codegenVersion
    )

  private def copyModule(
      compilerOptions: CompilerOptions = CompilerOptions()
  ): GeneratedCudaModule =
    val source = input[Float]("source")
    val destination = output[Float]("destination")
    val definition = kernel(
      "copyValues",
      params(source, destination)
    ) { bindings =>
      bindings.tail.head(threadIdx.x) :=
        bindings.head(threadIdx.x).read
    }

    CudaCodegen
      .generateModule(
        module(kernels = Vector(definition)),
        compilerOptions
      )
      .toOption
      .get

  private def scalarModule(): GeneratedCudaModule =
    val scalar = value[Float]("scalar")
    val destination = output[Float]("destination")
    val definition = kernel(
      "fillValues",
      params(scalar, destination)
    ) { bindings =>
      bindings.tail.head(threadIdx.x) := bindings.head
    }

    CudaCodegen
      .generateModule(module(kernels = Vector(definition)))
      .toOption
      .get

  private def withLaunchRequirements[Args <: Tuple](
      kernel: GeneratedKernel[Args],
      requirements: KernelLaunchRequirements
  ): GeneratedKernel[Args] =
    kernel.copy(launchRequirements = requirements)

  private def withSourceMetadata[Args <: Tuple](
      kernel: GeneratedKernel[Args],
      sourceMap: SourceMap
  ): GeneratedKernel[Args] =
    kernel.copy(
      sourceMap = sourceMap,
      declarationLine = kernel.declarationLine + 100
    )
