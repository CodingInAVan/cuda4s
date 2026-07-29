package com.cuda4s.core.ir

import com.cuda4s.core.types.CudaType

enum BinaryOperator(val cudaToken: String):
  case Add extends BinaryOperator("+")
  case Subtract extends BinaryOperator("-")
  case Multiply extends BinaryOperator("*")
  case Divide extends BinaryOperator("/")
  case Remainder extends BinaryOperator("%")

enum ComparisonOperator(val cudaToken: String):
  case LessThan extends ComparisonOperator("<")
  case LessThanOrEqual extends ComparisonOperator("<=")
  case GreaterThan extends ComparisonOperator(">")
  case GreaterThanOrEqual extends ComparisonOperator(">=")
  case Equal extends ComparisonOperator("==")
  case NotEqual extends ComparisonOperator("!=")

sealed trait Expr[T]:
  def valueType: CudaType[T]
  def span: SourceSpan

final case class Literal[T](
    value: T,
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[T]

final case class Binary[T](
    operator: BinaryOperator,
    left: Expr[T],
    right: Expr[T],
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[T]

final case class Compare[T](
    operator: ComparisonOperator,
    left: Expr[T],
    right: Expr[T],
    operandType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[Boolean]:
  override val valueType: CudaType[Boolean] = com.cuda4s.core.types.Bool

final case class Intrinsic[T](
    name: String,
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[T]

sealed trait AddressSpace
sealed trait Global extends AddressSpace
sealed trait Shared extends AddressSpace
sealed trait Local extends AddressSpace
sealed trait Constant extends AddressSpace

sealed trait AccessMode
sealed trait ReadOnly extends AccessMode
sealed trait ReadWrite extends AccessMode

sealed trait Place[T, Space <: AddressSpace, Mode <: AccessMode]:
  def valueType: CudaType[T]
  def span: SourceSpan

final case class BufferElement[T, Mode <: AccessMode](
    bufferName: String,
    index: Expr[Int],
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Place[T, Global, Mode]

final case class LocalVariable[T](
    name: String,
    valueType: CudaType[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Place[T, Local, ReadWrite]

final case class Load[
    T,
    Space <: AddressSpace,
    Mode <: AccessMode
](
    from: Place[T, Space, Mode],
    span: SourceSpan = SourceSpan.Unknown
) extends Expr[T]:
  override def valueType: CudaType[T] = from.valueType

sealed trait Stmt:
  def span: SourceSpan

final case class Store[T, Space <: AddressSpace](
    to: Place[T, Space, ReadWrite],
    value: Expr[T],
    span: SourceSpan = SourceSpan.Unknown
) extends Stmt

final case class IfThen(
    condition: Expr[Boolean],
    thenBlock: Block,
    elseBlock: Option[Block] = None,
    span: SourceSpan = SourceSpan.Unknown
) extends Stmt

final case class Barrier(
    span: SourceSpan = SourceSpan.Unknown
) extends Stmt

final case class Block(statements: Vector[Stmt])

sealed trait KernelParam:
  type Value

  def name: String
  def valueType: CudaType[Value]

final case class ScalarParam[T](
    name: String,
    valueType: CudaType[T]
) extends KernelParam:
  override type Value = T

final case class BufferParam[T, Mode <: AccessMode](
    name: String,
    valueType: CudaType[T]
) extends KernelParam:
  override type Value = T

final case class KernelIR(
    name: String,
    params: Vector[KernelParam],
    body: Block
)
