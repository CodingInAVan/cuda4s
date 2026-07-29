package flight4s.core.abi

import java.nio.ByteOrder

import munit.FunSuite

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*

class NativeArgumentStorageSuite extends FunSuite:
  test("native ABI codes are explicit and unique"):
    val expected = Map[CudaAbiType, Byte](
      CudaAbiType.Bool8 -> 1,
      CudaAbiType.SignedInt32 -> 2,
      CudaAbiType.UnsignedInt32 -> 3,
      CudaAbiType.Float16 -> 4,
      CudaAbiType.BFloat16 -> 5,
      CudaAbiType.Float32 -> 6,
      CudaAbiType.Float64 -> 7,
      CudaAbiType.Float8E4M3 -> 8,
      CudaAbiType.Float8E5M2 -> 9,
      CudaAbiType.DevicePointer64 -> 10
    )
    val actual = CudaAbiType.values.map(abi => abi -> abi.nativeCode).toMap

    assertEquals(actual, expected)
    assertEquals(
      CudaAbiType.values.map(_.nativeCode).distinct.length,
      CudaAbiType.values.length
    )

  test("arguments occupy aligned slots in one direct buffer"):
    val signature = params(
      value[Boolean]("enabled"),
      value[Double]("epsilon"),
      value[Int]("count"),
      in[Float]("input")
    )
    val definition = kernel("alignedArguments", signature) { _ => () }
    val native = definition
      .bind(
        (
          true,
          0.0001d,
          37,
          TestDeviceBuffer[Float](0x1122334455667788L)
        )
      )
      .nativeArguments

    assertEquals(native.offsets, Vector(0, 8, 16, 24))
    assertEquals(native.sizeBytes, 32)
    assertEquals(
      native.descriptorCodes,
      Vector(
        CudaAbiType.Bool8.nativeCode,
        CudaAbiType.Float64.nativeCode,
        CudaAbiType.SignedInt32.nativeCode,
        CudaAbiType.DevicePointer64.nativeCode
      )
    )
    assert(native.bufferForLaunch.isDirect)
    assertEquals(native.bufferForLaunch.alignmentOffset(0, 8), 0)

    val buffer = native.bufferForLaunch
    assertEquals(buffer.get(0), 1.toByte)
    assertEquals(buffer.getDouble(8), 0.0001d)
    assertEquals(buffer.getInt(16), 37)
    assertEquals(buffer.getLong(24), 0x1122334455667788L)

  test("empty signatures produce empty direct storage"):
    val definition = kernel("empty", params()) { _ => () }
    val native = definition.bind(EmptyTuple).nativeArguments

    assertEquals(native.slots, Vector.empty)
    assertEquals(native.offsets, Vector.empty)
    assertEquals(native.descriptorCodes, Vector.empty)
    assertEquals(native.sizeBytes, 0)
    assert(native.bufferForLaunch.isDirect)

  test("launch buffer views retain native order and independent positions"):
    val definition = kernel(
      "views",
      params(value[Int]("count"), value[Float]("scale"))
    ) { _ => () }
    val native = definition.bind((12, 1.5f)).nativeArguments
    val first = native.bufferForLaunch
    val second = native.bufferForLaunch

    first.position(4)

    assertEquals(first.order(), ByteOrder.nativeOrder())
    assertEquals(second.order(), ByteOrder.nativeOrder())
    assertEquals(second.position(), 0)
    assertEquals(second.getInt(0), 12)
    assertEquals(second.getFloat(4), 1.5f)

  private final case class TestDeviceBuffer[T](rawAddress: Long)
      extends DeviceBuffer[T]:
    override private[flight4s] val deviceAddress: DeviceAddress =
      DeviceAddress.fromRaw(rawAddress)
