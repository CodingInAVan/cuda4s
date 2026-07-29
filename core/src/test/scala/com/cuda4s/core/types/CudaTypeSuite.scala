package com.cuda4s.core.types

import munit.FunSuite

class CudaTypeSuite extends FunSuite:
  test("scalar CUDA types expose ABI metadata"):
    assertEquals(I32.cudaName, "int")
    assertEquals(I32.sizeBytes, 4)
    assertEquals(F16.cudaName, "__half")
    assertEquals(F16.sizeBytes, 2)
    assertEquals(F16.requiredHeaders, Set("cuda_fp16.h"))
    assertEquals(BF16.cudaName, "__nv_bfloat16")
    assertEquals(BF16.requiredHeaders, Set("cuda_bf16.h"))
    assertEquals(F32.cudaName, "float")
    assertEquals(F64.sizeBytes, 8)
    assertEquals(U32.cudaName, "unsigned int")
    assertEquals(FP8E4M3.cudaName, "__nv_fp8_e4m3")
    assertEquals(FP8E4M3.sizeBytes, 1)
    assertEquals(FP8E4M3.requiredHeaders, Set("cuda_fp8.h"))
    assertEquals(FP8E5M2.cudaName, "__nv_fp8_e5m2")

  test("UInt preserves unsigned integer bits"):
    val value = UInt.fromBits(-1)
    assertEquals(value.toIntBits, -1)

  test("low-precision host values preserve their raw bits"):
    val float16 = Float16.fromBits(0x3c00.toShort)
    val bfloat16 = BFloat16.fromBits(0x3f80.toShort)
    val e4m3 = Float8E4M3.fromBits(0x38.toByte)
    val e5m2 = Float8E5M2.fromBits(0x3c.toByte)

    assertEquals(float16.toShortBits, 0x3c00.toShort)
    assertEquals(bfloat16.toShortBits, 0x3f80.toShort)
    assertEquals(e4m3.toByteBits, 0x38.toByte)
    assertEquals(e5m2.toByteBits, 0x3c.toByte)

  test("operation capabilities are narrower than CUDA storage types"):
    assertEquals(summon[AdditiveType[Float]], F32)
    assertEquals(summon[RemainderType[Int]], I32)
    assertEquals(summon[OrderedType[Float16]], F16)

  test("default accumulation types widen low-precision values"):
    assertEquals(summon[AccumulatorType[Float16, Float]].inputType, F16)
    assertEquals(summon[AccumulatorType[Float16, Float]].accumulatorType, F32)
    assertEquals(summon[AccumulatorType[BFloat16, Float]].accumulatorType, F32)
    assertEquals(summon[AccumulatorType[Float8E4M3, Float]].accumulatorType, F32)
    assertEquals(summon[AccumulatorType[Float8E5M2, Float]].accumulatorType, F32)
    assertEquals(summon[AccumulatorType[Double, Double]].accumulatorType, F64)
