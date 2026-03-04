package com.cuda4s.jit.runtime

import com.cuda4s.jit.ir.{Block, KernelDef}
import munit.FunSuite
import com.cuda4s.jit.ir.*

class CacheSpec extends FunSuite {
  test("KernelCache hit and miss") {
    var compileCount = 0
    val cache = new KernelCache((src, name) => {
      compileCount += 1
      compileCount.toLong
    })
    
    val k1 = KernelDef("k1", Nil, Block(Nil))
    val src1 = "src1"
    
    val handle1 = cache.getOrCompile(k1, src1)
    assertEquals(handle1, 1L)
    assertEquals(compileCount, 1)
    
    // Hit
    val handle2 = cache.getOrCompile(k1, src1)
    assertEquals(handle2, 1L)
    assertEquals(compileCount, 1)
    
    // Miss due to source change
    val handle3 = cache.getOrCompile(k1, "src2")
    assertEquals(handle3, 2L)
    assertEquals(compileCount, 2)
  }

  test("KernelCache string-based hit and miss") {
    var compileCount = 0
    val cache = new KernelCache((src, name) => {
      compileCount += 1
      compileCount.toLong
    })

    val handle1 = cache.getOrCompile("myKernel", "src1")
    assertEquals(handle1, 1L)
    assertEquals(compileCount, 1)

    // Hit
    val handle2 = cache.getOrCompile("myKernel", "src1")
    assertEquals(handle2, 1L)
    assertEquals(compileCount, 1)

    // Miss due to source change
    val handle3 = cache.getOrCompile("myKernel", "src2")
    assertEquals(handle3, 2L)
    assertEquals(compileCount, 2)
  }

  test("KernelCache closeAll") {
    var closedHandles = List.empty[Long]
    val cache = new KernelCache((src, name) => 10L)
    cache.getOrCompile(KernelDef("k", Nil, Block(Nil)), "src")
    
    cache.closeAll(h => closedHandles = h :: closedHandles)
    assertEquals(closedHandles, List(10L))
  }
}
