package com.cuda4s.core.dsl

import scala.collection.mutable.ArrayBuffer

import com.cuda4s.core.ir.*
import com.cuda4s.core.types.*

object CudaDsl:
  final class BlockBuilder private[dsl] ():
    private val statements = ArrayBuffer.empty[Stmt]

    private[dsl] def append(statement: Stmt): Unit =
      statements += statement

    private[dsl] def result(): Block =
      Block(statements.toVector)

  def literal[T](value: T)(using valueType: CudaType[T]): Expr[T] =
    Literal(value, valueType)

  def input[T](name: String)(using valueType: CudaType[T]): BufferParam[T, ReadOnly] =
    BufferParam(name, valueType)

  def output[T](name: String)(using valueType: CudaType[T]): BufferParam[T, ReadWrite] =
    BufferParam(name, valueType)

  def inOut[T](name: String)(using valueType: CudaType[T]): BufferParam[T, ReadWrite] =
    BufferParam(name, valueType)

  def value[T](name: String)(using valueType: CudaType[T]): ScalarParam[T] =
    ScalarParam(name, valueType)

  def kernel(
      name: String,
      params: KernelParam*
  )(body: BlockBuilder ?=> Unit): KernelIR =
    val builder = BlockBuilder()
    body(using builder)
    KernelIR(name, params.toVector, builder.result())

  def when(condition: Expr[Boolean])(
      body: BlockBuilder ?=> Unit
  )(using parent: BlockBuilder): Unit =
    val nested = BlockBuilder()
    body(using nested)
    parent.append(IfThen(condition, nested.result()))

  def gpuIf(condition: Expr[Boolean])(
      thenBody: BlockBuilder ?=> Unit
  )(
      elseBody: BlockBuilder ?=> Unit
  )(using parent: BlockBuilder): Unit =
    val thenBuilder = BlockBuilder()
    thenBody(using thenBuilder)
    val elseBuilder = BlockBuilder()
    elseBody(using elseBuilder)
    parent.append(
      IfThen(
        condition = condition,
        thenBlock = thenBuilder.result(),
        elseBlock = Some(elseBuilder.result())
      )
    )

  def barrier()(using builder: BlockBuilder): Unit =
    builder.append(Barrier())

  object threadIdx:
    def x: Expr[Int] = Intrinsic("threadIdx.x", I32)
    def y: Expr[Int] = Intrinsic("threadIdx.y", I32)
    def z: Expr[Int] = Intrinsic("threadIdx.z", I32)

  object blockIdx:
    def x: Expr[Int] = Intrinsic("blockIdx.x", I32)
    def y: Expr[Int] = Intrinsic("blockIdx.y", I32)
    def z: Expr[Int] = Intrinsic("blockIdx.z", I32)

  object blockDim:
    def x: Expr[Int] = Intrinsic("blockDim.x", I32)
    def y: Expr[Int] = Intrinsic("blockDim.y", I32)
    def z: Expr[Int] = Intrinsic("blockDim.z", I32)

  extension [T](left: Expr[T])
    def +(right: Expr[T])(using valueType: AdditiveType[T]): Expr[T] =
      Binary(BinaryOperator.Add, left, right, valueType)

    def -(right: Expr[T])(using valueType: AdditiveType[T]): Expr[T] =
      Binary(BinaryOperator.Subtract, left, right, valueType)

    def *(right: Expr[T])(using valueType: MultiplicativeType[T]): Expr[T] =
      Binary(BinaryOperator.Multiply, left, right, valueType)

    def /(right: Expr[T])(using valueType: DivisibleType[T]): Expr[T] =
      Binary(BinaryOperator.Divide, left, right, valueType)

    def %(right: Expr[T])(using valueType: RemainderType[T]): Expr[T] =
      Binary(BinaryOperator.Remainder, left, right, valueType)

    def <(right: Expr[T])(using valueType: OrderedType[T]): Expr[Boolean] =
      Compare(ComparisonOperator.LessThan, left, right, valueType)

    def <=(right: Expr[T])(using valueType: OrderedType[T]): Expr[Boolean] =
      Compare(ComparisonOperator.LessThanOrEqual, left, right, valueType)

    def >(right: Expr[T])(using valueType: OrderedType[T]): Expr[Boolean] =
      Compare(ComparisonOperator.GreaterThan, left, right, valueType)

    def >=(right: Expr[T])(using valueType: OrderedType[T]): Expr[Boolean] =
      Compare(ComparisonOperator.GreaterThanOrEqual, left, right, valueType)

    def ===(right: Expr[T])(using valueType: EqualityComparableType[T]): Expr[Boolean] =
      Compare(ComparisonOperator.Equal, left, right, valueType)

    def !==(right: Expr[T])(using valueType: EqualityComparableType[T]): Expr[Boolean] =
      Compare(ComparisonOperator.NotEqual, left, right, valueType)

    def toAccumulator[A](using rule: AccumulatorType[T, A]): Expr[A] =
      ToAccumulator(left, rule)

  object convert:
    def f32ToF16(
        value: Expr[Float],
        rounding: RoundingMode = RoundingMode.NearestEven
    ): Expr[Float16] =
      Convert(
        value = value,
        valueType = F16,
        rounding = rounding,
        saturation = SaturationMode.NoSaturation
      )

    def f16ToF32(value: Expr[Float16]): Expr[Float] =
      Convert(
        value = value,
        valueType = F32,
        rounding = RoundingMode.NearestEven,
        saturation = SaturationMode.NoSaturation
      )

    def f32ToBF16(
        value: Expr[Float],
        rounding: RoundingMode = RoundingMode.NearestEven
    ): Expr[BFloat16] =
      Convert(
        value = value,
        valueType = BF16,
        rounding = rounding,
        saturation = SaturationMode.NoSaturation
      )

    def bf16ToF32(value: Expr[BFloat16]): Expr[Float] =
      Convert(
        value = value,
        valueType = F32,
        rounding = RoundingMode.NearestEven,
        saturation = SaturationMode.NoSaturation
      )

    def f32ToFP8E4M3(
        value: Expr[Float],
        saturation: SaturationMode = SaturationMode.SaturateFinite
    ): Expr[Float8E4M3] =
      Convert(
        value = value,
        valueType = FP8E4M3,
        rounding = RoundingMode.NearestEven,
        saturation = saturation
      )

    def fp8E4M3ToF32(value: Expr[Float8E4M3]): Expr[Float] =
      Convert(
        value = value,
        valueType = F32,
        rounding = RoundingMode.NearestEven,
        saturation = SaturationMode.NoSaturation
      )

    def f32ToFP8E5M2(
        value: Expr[Float],
        saturation: SaturationMode = SaturationMode.SaturateFinite
    ): Expr[Float8E5M2] =
      Convert(
        value = value,
        valueType = FP8E5M2,
        rounding = RoundingMode.NearestEven,
        saturation = saturation
      )

    def fp8E5M2ToF32(value: Expr[Float8E5M2]): Expr[Float] =
      Convert(
        value = value,
        valueType = F32,
        rounding = RoundingMode.NearestEven,
        saturation = SaturationMode.NoSaturation
      )

  extension [T, Mode <: AccessMode](buffer: BufferParam[T, Mode])
    def apply(index: Expr[Int]): BufferElement[T, Mode] =
      BufferElement(buffer.name, index, buffer.valueType)

  extension [T, Space <: AddressSpace, Mode <: AccessMode](
      place: Place[T, Space, Mode]
  )
    def read: Expr[T] =
      Load(place)

  extension [T, Space <: AddressSpace](
      place: Place[T, Space, ReadWrite]
  )
    infix def :=(value: Expr[T])(using builder: BlockBuilder): Unit =
      builder.append(Store(place, value))
