package flight4s.core.ir

trait DeviceBuffer[T]

type KernelArgumentOf[Param] = Param match
  case ScalarParam[value] => value
  case BufferParam[value, mode] => DeviceBuffer[value]

type KernelArgumentsOf[Params <: Tuple] <: Tuple = Params match
  case EmptyTuple => EmptyTuple
  case head *: tail => KernelArgumentOf[head] *: KernelArgumentsOf[tail]

sealed trait KernelParamTuple[Params <: Tuple]:
  def toVector(params: Params): Vector[KernelParam]

object KernelParamTuple:
  given emptyTuple: KernelParamTuple[EmptyTuple] with
    override def toVector(params: EmptyTuple): Vector[KernelParam] =
      Vector.empty

  given nonEmptyTuple[
      Head <: KernelParam,
      Tail <: Tuple
  ](using tailParams: KernelParamTuple[Tail]): KernelParamTuple[Head *: Tail] with
    override def toVector(params: Head *: Tail): Vector[KernelParam] =
      params.head +: tailParams.toVector(params.tail)

sealed trait KernelSignature[Args <: Tuple]:
  type Bindings <: Tuple

  def bindings: Bindings
  def parameters: Vector[KernelParam]

object KernelSignature:
  private final class Impl[Params <: Tuple](
      override val bindings: Params,
      tuple: KernelParamTuple[Params]
  ) extends KernelSignature[KernelArgumentsOf[Params]]:
    override type Bindings = Params
    override val parameters: Vector[KernelParam] = tuple.toVector(bindings)

  def fromTuple[Params <: Tuple](
      bindings: Params
  )(using tuple: KernelParamTuple[Params])
      : KernelSignature[KernelArgumentsOf[Params]] { type Bindings = Params } =
    new Impl(bindings, tuple)

final case class KernelIR[Args <: Tuple](
    name: String,
    signature: KernelSignature[Args],
    body: Block
):
  def params: Vector[KernelParam] = signature.parameters

final case class Kernel[Args <: Tuple](ir: KernelIR[Args]):
  def name: String = ir.name
  def signature: KernelSignature[Args] = ir.signature
  def params: Vector[KernelParam] = ir.params
  def body: Block = ir.body

  def bind(arguments: Args): KernelInvocation[Args] =
    KernelInvocation(this, arguments)

final case class KernelInvocation[Args <: Tuple](
    kernel: Kernel[Args],
    arguments: Args
)
