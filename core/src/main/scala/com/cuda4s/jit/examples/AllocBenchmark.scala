package com.cuda4s.jit.examples

import com.cuda4s.CudaJitJNI
import com.cuda4s.jit.utils.BenchmarkUtils.timeMs

object AllocBenchmark:
  def main(args: Array[String]): Unit =
    val jni = CudaJitJNI.instance
    val numAllocs = 1000
    val size = 1024L

    // Warmup
    for (_ <- 0 until 100) {
      val p = jni.alloc(size)
      jni.free(p)
    }

    val ptrs = new Array[Long](numAllocs)
    val individualTime = timeMs(s"Individual allocs (x$numAllocs)", warmup = 0, iters = 1) {
      for (i <- 0 until numAllocs) {
        ptrs(i) = jni.alloc(size)
      }
      for (i <- 0 until numAllocs) {
        jni.free(ptrs(i))
      }
      0.0
    }

    val batchTime = timeMs(s"Batch allocs      (x$numAllocs)", warmup = 0, iters = 1) {
      val sizes = Array.fill(numAllocs)(size)
      val ptrsBatch = jni.allocMany(sizes)
      jni.freeMany(ptrsBatch)
      0.0
    }
    
    println(f"Speedup: ${individualTime / batchTime}%.2fx")
