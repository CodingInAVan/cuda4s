package flight4s.runtime.cuda

import java.nio.ByteOrder

import munit.FunSuite

import flight4s.core.types.*

class CudaHostCodecSuite extends FunSuite:
  test("host codecs round-trip every supported CUDA scalar type"):
    assertRoundTrip(Array(false, true, true, false))
    assertRoundTrip(Array(Int.MinValue, -1, 0, Int.MaxValue))
    assertRoundTrip(
      Array(
        UInt.fromBits(Int.MinValue),
        UInt.fromBits(0),
        UInt.fromBits(Int.MaxValue)
      )
    )
    assertRoundTrip(
      Array(
        Float16.fromBits(Short.MinValue),
        Float16.fromBits(0),
        Float16.fromBits(Short.MaxValue)
      )
    )
    assertRoundTrip(
      Array(
        BFloat16.fromBits(Short.MinValue),
        BFloat16.fromBits(0),
        BFloat16.fromBits(Short.MaxValue)
      )
    )
    assertRoundTrip(
      Array(
        Float.NegativeInfinity,
        -0.0f,
        1.25f,
        Float.PositiveInfinity
      )
    )
    assertRoundTrip(
      Array(
        Double.NegativeInfinity,
        -0.0d,
        1.25d,
        Double.PositiveInfinity
      )
    )
    assertRoundTrip(
      Array(
        Float8E4M3.fromBits(Byte.MinValue),
        Float8E4M3.fromBits(0),
        Float8E4M3.fromBits(Byte.MaxValue)
      )
    )
    assertRoundTrip(
      Array(
        Float8E5M2.fromBits(Byte.MinValue),
        Float8E5M2.fromBits(0),
        Float8E5M2.fromBits(Byte.MaxValue)
      )
    )

  test("encoded host storage is an exact native-order direct view"):
    val codec = summon[CudaHostCodec[Int]]
    val bytes = codec.encode(Array(0x01020304, 0x11223344))

    assert(bytes.isDirect)
    assertEquals(bytes.position(), 0)
    assertEquals(bytes.limit(), 8)
    assertEquals(bytes.capacity(), 8)
    assertEquals(bytes.order(), ByteOrder.nativeOrder())
    assertEquals(
      bytes.duplicate().order(ByteOrder.nativeOrder()).getInt(),
      0x01020304
    )

  private def assertRoundTrip[T](
      values: Array[T]
  )(using codec: CudaHostCodec[T]): Unit =
    val bytes = codec.encode(values)
    val decoded = codec.decode(bytes, values.length)

    assert(
      decoded.sameElements(values),
      s"${codec.cudaType.cudaName} host values did not round-trip"
    )
