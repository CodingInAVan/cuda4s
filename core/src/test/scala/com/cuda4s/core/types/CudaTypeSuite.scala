package com.cuda4s.core.types

import munit.FunSuite

class CudaTypeSuite extends FunSuite:
  test("scalar CUDA types expose ABI metadata"):
    assertEquals(I32.cudaName, "int")
    assertEquals(I32.sizeBytes, 4)
    assertEquals(F32.cudaName, "float")
    assertEquals(F64.sizeBytes, 8)
    assertEquals(U32.cudaName, "unsigned int")

  test("UInt preserves unsigned integer bits"):
    val value = UInt.fromBits(-1)
    assertEquals(value.toIntBits, -1)
