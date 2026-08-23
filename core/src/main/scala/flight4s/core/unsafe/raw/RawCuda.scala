package flight4s.core.unsafe.raw

import flight4s.core.abi.{KernelArgumentAbi, PackedKernelArguments}
import flight4s.core.codegen.{CompilerOptions, KernelLaunchRequirements}
import flight4s.core.ir.{KernelParam, KernelSignature}

enum RawCudaDefinitionErrorCode:
  case InvalidEntryPoint
  case EmptySource
  case SourceContainsNullCharacter
  case InvalidParameterName
  case DuplicateParameterName

final case class RawCudaDefinitionError(
    code: RawCudaDefinitionErrorCode,
    message: String
) extends IllegalArgumentException(message)

/**
 * A caller-provided CUDA C++ kernel definition.
 *
 * Flight4s validates the surrounding metadata but does not parse the source or
 * prove that its entry point and parameter ABI agree with the typed signature.
 */
final class RawCudaKernel[Args <: Tuple] private[raw] (
    val entryPoint: String,
    val signature: KernelSignature[Args],
    val source: String,
    val compilerOptions: CompilerOptions,
    val launchRequirements: KernelLaunchRequirements
):
  def parameters: Vector[KernelParam] = signature.parameters

  def abiDescriptors: Vector[KernelArgumentAbi] = signature.abiDescriptors

  def bind(arguments: Args): RawCudaInvocation[Args] =
    RawCudaInvocation(this, arguments)

  override def toString: String = s"RawCudaKernel($entryPoint)"

final case class RawCudaInvocation[Args <: Tuple](
    kernel: RawCudaKernel[Args],
    arguments: Args
):
  private[flight4s] def packedArguments: PackedKernelArguments =
    kernel.signature.pack(arguments)

object RawCuda:
  private val cudaIdentifier = raw"[A-Za-z_][A-Za-z0-9_]*".r

  def kernel[Args <: Tuple](
      entryPoint: String,
      signature: KernelSignature[Args],
      source: String,
      compilerOptions: CompilerOptions,
      launchRequirements: KernelLaunchRequirements
  ): RawCudaKernel[Args] =
    validateEntryPoint(entryPoint)
    validateSource(source)
    validateParameters(signature.parameters)

    new RawCudaKernel(
      entryPoint,
      signature,
      source,
      compilerOptions,
      launchRequirements
    )

  private def validateEntryPoint(entryPoint: String): Unit =
    if entryPoint == null || !cudaIdentifier.matches(entryPoint) then
      throw RawCudaDefinitionError(
        RawCudaDefinitionErrorCode.InvalidEntryPoint,
        s"'$entryPoint' is not a valid CUDA entry-point identifier"
      )

  private def validateSource(source: String): Unit =
    if source == null || !source.exists(character => !character.isWhitespace) then
      throw RawCudaDefinitionError(
        RawCudaDefinitionErrorCode.EmptySource,
        "raw CUDA source must contain non-whitespace text"
      )
    if source.contains('\u0000') then
      throw RawCudaDefinitionError(
        RawCudaDefinitionErrorCode.SourceContainsNullCharacter,
        "raw CUDA source must not contain a null character"
      )

  private def validateParameters(parameters: Vector[KernelParam]): Unit =
    parameters.zipWithIndex.find { case (parameter, _) =>
      parameter.name == null || !cudaIdentifier.matches(parameter.name)
    }.foreach { case (parameter, index) =>
      throw RawCudaDefinitionError(
        RawCudaDefinitionErrorCode.InvalidParameterName,
        s"raw CUDA parameter at index $index has invalid identifier '${parameter.name}'"
      )
    }

    parameters
      .groupBy(_.name)
      .toVector
      .collect { case (name, declarations) if declarations.size > 1 => name }
      .sorted
      .headOption
      .foreach { name =>
        throw RawCudaDefinitionError(
          RawCudaDefinitionErrorCode.DuplicateParameterName,
          s"raw CUDA parameter '$name' is declared more than once"
        )
      }
