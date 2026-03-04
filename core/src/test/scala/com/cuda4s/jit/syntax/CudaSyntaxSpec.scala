package com.cuda4s.jit.syntax

import munit.FunSuite
import com.cuda4s.jit.syntax.CudaSyntax.*

class CudaSyntaxSpec extends FunSuite {
  test("basic interpolation") {
    val n = 256
    val src = cuda"int x = $n;"
    assertEquals(src, "int x = 256;")
  }

  test("stripMargin support") {
    val src = cuda"""
      |line1
      |line2
      """
    assert(src.contains("line1"))
    assert(src.contains("line2"))
    assert(!src.contains("|"))
  }

  test("multiple interpolated values") {
    val name = "vecAdd"
    val blockSize = 256
    val src = cuda"""
      |extern "C" __global__
      |void $name(float* a, int n) {
      |  int i = threadIdx.x + blockIdx.x * $blockSize;
      |}
      """
    assert(src.contains("void vecAdd"))
    assert(src.contains("blockIdx.x * 256"))
  }

  test("no interpolation returns plain string") {
    val src = cuda"""
      |__global__ void k() {}
      """
    assert(src.contains("__global__ void k()"))
  }
}
