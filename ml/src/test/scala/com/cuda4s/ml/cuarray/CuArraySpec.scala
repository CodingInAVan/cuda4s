package com.cuda4s.ml.cuarray

import com.cuda4s.jit.runtime.GpuBackend
import munit.FunSuite

class CuArraySpec extends FunSuite:
  given backend: GpuBackend = new GpuBackend()

  test("CuArray arithmetic") {
    val a = CuArray(Array(1.0f, 2.0f, 3.0f, 4.0f), 2, 2)
    val b = CuArray(Array(5.0f, 6.0f, 7.0f, 8.0f), 2, 2)
    
    val resAdd = (a + b)
    val h_resAdd = new Array[Float](4)
    resAdd.copyToHost(h_resAdd)
    assert(h_resAdd.sameElements(Array(6.0f, 8.0f, 10.0f, 12.0f)))

    val resSub = (a - b)
    val h_resSub = new Array[Float](4)
    resSub.copyToHost(h_resSub)
    assert(h_resSub.sameElements(Array(-4.0f, -4.0f, -4.0f, -4.0f)))

    val resMul = (a * b)
    val h_resMul = new Array[Float](4)
    resMul.copyToHost(h_resMul)
    assert(h_resMul.sameElements(Array(5.0f, 12.0f, 21.0f, 32.0f)))

    val resDiv = (b / a)
    val h_resDiv = new Array[Float](4)
    resDiv.copyToHost(h_resDiv)
    assert(h_resDiv.sameElements(Array(5.0f, 3.0f, 7.0f/3.0f, 2.0f)))
  }

  test("CuArray sum and mean") {
    val a = CuArray(Array(1.0f, 2.0f, 3.0f, 4.0f), 4)
    assertEquals(a.sum, 10.0f)
    assertEquals(a.mean, 2.5f)
  }

  test("CuArray relu") {
    val a = CuArray(Array(-1.0f, 0.0f, 1.0f))
    val b = a.relu
    val res = new Array[Float](3)
    b.copyToHost(res)
    assert(res.sameElements(Array(0.0f, 0.0f, 1.0f)))
  }

  test("CuArray multidimensional shape") {
    val a = CuArray.ones(2, 3)
    assertEquals(a.shape, Seq(2, 3))
    assertEquals(a.size, 6)
    
    val res = new Array[Float](6)
    a.copyToHost(res)
    assert(res.forall(_ == 1.0f))
  }

  test("CuArray zeros and fill") {
    val a = CuArray.zeros(10)
    val res = new Array[Float](10)
    a.copyToHost(res)
    assert(res.forall(_ == 0.0f))
    
    val b = CuArray.fill(7.0f, 2, 2)
    val res2 = new Array[Float](4)
    b.copyToHost(res2)
    assert(res2.forall(_ == 7.0f))
  }
