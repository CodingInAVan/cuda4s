package flight4s.core.abi

import java.nio.{ByteBuffer, ByteOrder}

import flight4s.core.types.*

enum CudaAbiType(
    val nativeCode: Byte,
    val sizeBytes: Int,
    val alignmentBytes: Int
):
  case Bool8 extends CudaAbiType(1, 1, 1)
  case SignedInt32 extends CudaAbiType(2, 4, 4)
  case UnsignedInt32 extends CudaAbiType(3, 4, 4)
  case Float16 extends CudaAbiType(4, 2, 2)
  case BFloat16 extends CudaAbiType(5, 2, 2)
  case Float32 extends CudaAbiType(6, 4, 4)
  case Float64 extends CudaAbiType(7, 8, 8)
  case Float8E4M3 extends CudaAbiType(8, 1, 1)
  case Float8E5M2 extends CudaAbiType(9, 1, 1)
  case DevicePointer64 extends CudaAbiType(10, 8, 8)

final case class KernelArgumentAbi(
    name: String,
    abiType: CudaAbiType
)

final case class PackedKernelArgument(
    descriptor: KernelArgumentAbi,
    bytes: Vector[Byte]
):
  require(
    bytes.size == descriptor.abiType.sizeBytes,
    s"${descriptor.name} requires ${descriptor.abiType.sizeBytes} bytes, " +
      s"but received ${bytes.size}"
  )

final case class PackedKernelArguments(
    arguments: Vector[PackedKernelArgument]
):
  def descriptors: Vector[KernelArgumentAbi] =
    arguments.map(_.descriptor)

opaque type DeviceAddress = Long

object DeviceAddress:
  private[flight4s] def fromRaw(raw: Long): DeviceAddress = raw

  private[flight4s] def encode(address: DeviceAddress): Vector[Byte] =
    NativeBytes.encode(java.lang.Long.BYTES)(_.putLong(address))

sealed trait ScalarAbi[T]:
  def abiType: CudaAbiType
  def encode(value: T): Vector[Byte]

object ScalarAbi:
  given boolAbi: ScalarAbi[Boolean] with
    override val abiType: CudaAbiType = CudaAbiType.Bool8
    override def encode(value: Boolean): Vector[Byte] =
      Vector(if value then 1.toByte else 0.toByte)

  given intAbi: ScalarAbi[Int] with
    override val abiType: CudaAbiType = CudaAbiType.SignedInt32
    override def encode(value: Int): Vector[Byte] =
      NativeBytes.encode(java.lang.Integer.BYTES)(_.putInt(value))

  given uintAbi: ScalarAbi[UInt] with
    override val abiType: CudaAbiType = CudaAbiType.UnsignedInt32
    override def encode(value: UInt): Vector[Byte] =
      NativeBytes.encode(java.lang.Integer.BYTES)(_.putInt(value.toIntBits))

  given float16Abi: ScalarAbi[Float16] with
    override val abiType: CudaAbiType = CudaAbiType.Float16
    override def encode(value: Float16): Vector[Byte] =
      NativeBytes.encode(java.lang.Short.BYTES)(_.putShort(value.toShortBits))

  given bfloat16Abi: ScalarAbi[BFloat16] with
    override val abiType: CudaAbiType = CudaAbiType.BFloat16
    override def encode(value: BFloat16): Vector[Byte] =
      NativeBytes.encode(java.lang.Short.BYTES)(_.putShort(value.toShortBits))

  given floatAbi: ScalarAbi[Float] with
    override val abiType: CudaAbiType = CudaAbiType.Float32
    override def encode(value: Float): Vector[Byte] =
      NativeBytes.encode(java.lang.Float.BYTES)(_.putFloat(value))

  given doubleAbi: ScalarAbi[Double] with
    override val abiType: CudaAbiType = CudaAbiType.Float64
    override def encode(value: Double): Vector[Byte] =
      NativeBytes.encode(java.lang.Double.BYTES)(_.putDouble(value))

  given float8E4M3Abi: ScalarAbi[Float8E4M3] with
    override val abiType: CudaAbiType = CudaAbiType.Float8E4M3
    override def encode(value: Float8E4M3): Vector[Byte] =
      Vector(value.toByteBits)

  given float8E5M2Abi: ScalarAbi[Float8E5M2] with
    override val abiType: CudaAbiType = CudaAbiType.Float8E5M2
    override def encode(value: Float8E5M2): Vector[Byte] =
      Vector(value.toByteBits)

private object NativeBytes:
  def encode(sizeBytes: Int)(write: ByteBuffer => Unit): Vector[Byte] =
    val buffer = ByteBuffer.allocate(sizeBytes).order(ByteOrder.nativeOrder())
    write(buffer)
    buffer.array().toVector
