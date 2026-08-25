package flight4s.core.compiler

import flight4s.core.abi.KernelArgumentAbi
import flight4s.core.codegen.{
  CompilerOptions,
  CudaCodegen,
  GeneratedCudaModule,
  KernelLaunchRequirements,
  SourceMap
}
import flight4s.core.ir.KernelSignature
import flight4s.core.unsafe.raw.RawCudaKernel

enum NvrtcSourceProvenance:
  case DslGenerated(codegenVersion: Int)
  case CallerProvidedRaw

final case class NvrtcKernelInput[Args <: Tuple](
    entryPoint: String,
    signature: KernelSignature[Args],
    launchRequirements: KernelLaunchRequirements
):
  def abiDescriptors: Vector[KernelArgumentAbi] = signature.abiDescriptors

sealed trait NvrtcCompilationInput:
  def provenance: NvrtcSourceProvenance
  def source: String
  def sourceMap: SourceMap
  def compilerOptions: CompilerOptions
  def kernels: Vector[NvrtcKernelInput[?]]

object NvrtcCompilationInput:
  final class Generated private[compiler] (
      val module: GeneratedCudaModule,
      val codegenVersion: Int
  ) extends NvrtcCompilationInput:
    override val provenance: NvrtcSourceProvenance =
      NvrtcSourceProvenance.DslGenerated(codegenVersion)
    override val source: String = module.cudaSource
    override val sourceMap: SourceMap = module.sourceMap
    override val compilerOptions: CompilerOptions = module.compilerOptions
    override val kernels: Vector[NvrtcKernelInput[?]] =
      module.kernels.map { kernel =>
        NvrtcKernelInput(
          kernel.name,
          kernel.signature,
          kernel.launchRequirements
        )
      }

  final class Raw[Args <: Tuple] private[compiler] (
      val definition: RawCudaKernel[Args]
  ) extends NvrtcCompilationInput:
    override val provenance: NvrtcSourceProvenance =
      NvrtcSourceProvenance.CallerProvidedRaw
    override val source: String = definition.source
    override val sourceMap: SourceMap = SourceMap(Vector.empty)
    override val compilerOptions: CompilerOptions = definition.compilerOptions
    override val kernels: Vector[NvrtcKernelInput[?]] =
      Vector(
        NvrtcKernelInput(
          definition.entryPoint,
          definition.signature,
          definition.launchRequirements
        )
      )

  def generated(
      module: GeneratedCudaModule,
      codegenVersion: Int = CudaCodegen.ArtifactVersion
  ): Generated =
    require(module != null, "generated CUDA module must not be null")
    require(codegenVersion > 0, "codegen version must be positive")
    new Generated(module, codegenVersion)

  def raw[Args <: Tuple](definition: RawCudaKernel[Args]): Raw[Args] =
    require(definition != null, "raw CUDA definition must not be null")
    new Raw(definition)
