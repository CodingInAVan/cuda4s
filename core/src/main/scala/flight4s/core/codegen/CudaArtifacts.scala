package flight4s.core.codegen

import flight4s.core.ir.{KernelSignature, SourceSpan, ValidationError}

enum CudaCppStandard(val nvrtcOption: String):
  case Cpp20 extends CudaCppStandard("--std=c++20")

final case class CompilerOptions(
    languageStandard: CudaCppStandard = CudaCppStandard.Cpp20,
    additionalNvrtcOptions: Vector[String] = Vector.empty
):
  require(
    additionalNvrtcOptions.forall(option => !option.startsWith("--std")),
    "additional NVRTC options cannot override the CUDA C++ language standard"
  )

  def nvrtcOptions: Vector[String] =
    languageStandard.nvrtcOption +: additionalNvrtcOptions

final case class SourceMapEntry(
    generatedLine: Int,
    sourceSpan: SourceSpan
)

final case class SourceMap(entries: Vector[SourceMapEntry]):
  def closestAtOrBefore(generatedLine: Int): Option[SourceMapEntry] =
    entries
      .filter(_.generatedLine <= generatedLine)
      .maxByOption(_.generatedLine)

final case class GeneratedKernel[Args <: Tuple](
    name: String,
    signature: KernelSignature[Args],
    cudaSource: String,
    sourceMap: SourceMap,
    compilerOptions: CompilerOptions,
    declarationLine: Int
)

final case class GeneratedCudaModule(
    cudaSource: String,
    sourceMap: SourceMap,
    compilerOptions: CompilerOptions,
    kernels: Vector[GeneratedKernel[?]]
)

sealed trait CodegenError:
  def message: String

object CodegenError:
  final case class ValidationFailed(errors: Vector[ValidationError])
      extends CodegenError:
    override val message: String =
      s"CUDA source generation rejected ${errors.size} validation error(s)"

  final case class UnsupportedLiteral(
      cudaType: String,
      valueClass: String,
      span: SourceSpan
  ) extends CodegenError:
    override val message: String =
      s"cannot emit a $cudaType literal from $valueClass"

  final case class UnsupportedConversion(
      fromType: String,
      toType: String,
      span: SourceSpan
  ) extends CodegenError:
    override val message: String =
      s"cannot emit a conversion from $fromType to $toType"
