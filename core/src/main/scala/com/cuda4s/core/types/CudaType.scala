package com.cuda4s.core.types

sealed trait CudaType[T]:
  def cudaName: String
  def sizeBytes: Int
  def alignmentBytes: Int

sealed trait NumericCudaType[T] extends CudaType[T]

final case class UInt private (bits: Int) extends AnyVal

object UInt:
  def fromBits(bits: Int): UInt = UInt(bits)

  extension (value: UInt)
    def toIntBits: Int = value.bits

case object Bool extends CudaType[Boolean]:
  override val cudaName: String = "bool"
  override val sizeBytes: Int = 1
  override val alignmentBytes: Int = 1

case object I32 extends NumericCudaType[Int]:
  override val cudaName: String = "int"
  override val sizeBytes: Int = 4
  override val alignmentBytes: Int = 4

case object U32 extends NumericCudaType[UInt]:
  override val cudaName: String = "unsigned int"
  override val sizeBytes: Int = 4
  override val alignmentBytes: Int = 4

case object F32 extends NumericCudaType[Float]:
  override val cudaName: String = "float"
  override val sizeBytes: Int = 4
  override val alignmentBytes: Int = 4

case object F64 extends NumericCudaType[Double]:
  override val cudaName: String = "double"
  override val sizeBytes: Int = 8
  override val alignmentBytes: Int = 8

object CudaType:
  given boolType: CudaType[Boolean] = Bool
  given intType: CudaType[Int] = I32
  given uintType: CudaType[UInt] = U32
  given floatType: CudaType[Float] = F32
  given doubleType: CudaType[Double] = F64

object NumericCudaType:
  given intType: NumericCudaType[Int] = I32
  given uintType: NumericCudaType[UInt] = U32
  given floatType: NumericCudaType[Float] = F32
  given doubleType: NumericCudaType[Double] = F64
