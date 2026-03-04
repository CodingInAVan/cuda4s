package com.cuda4s.ml.examples

import com.cuda4s.jit.runtime.GpuBackend
import com.cuda4s.ml.cuarray.CuArray
import java.util.concurrent.TimeUnit

object CuArrayVsArrayExample:
  def main(args: Array[String]): Unit =
    // 2^24 elements (~16.7M)
    val n = 1 << 24
    val shape = Seq(n)
    
    println(s"Benchmark: Array[Float] (CPU) vs CuArray (GPU)")
    println(s"Vector size: $n elements (${n * 4.0 / 1024 / 1024} MB per vector)")
    
    given backend: GpuBackend = new GpuBackend()
    
    // Initialize data
    val h_a = Array.fill(n)(1.0f)
    val h_b = Array.fill(n)(2.0f)
    
    // --- CPU Bench ---
    val startCpu = System.nanoTime()
    val h_res = new Array[Float](n)
    for (i <- 0 until n) {
      h_res(i) = h_a(i) + h_b(i)
      if (h_res(i) < 0) h_res(i) = 0
    }
    val endCpu = System.nanoTime()
    val cpuTimeMs = (endCpu - startCpu) / 1e6
    println(f"CPU time (Add + ReLU): $cpuTimeMs%.3f ms")
    
    // --- GPU Bench (including transfers) ---
    val startGpuTotal = System.nanoTime()
    val d_a = CuArray(h_a)
    val d_b = CuArray(h_b)
    
    val d_res = (d_a + d_b).relu
    
    val h_gpu_res = new Array[Float](n)
    d_res.copyToHost(h_gpu_res)
    val endGpuTotal = System.nanoTime()
    val gpuTotalTimeMs = (endGpuTotal - startGpuTotal) / 1e6
    println(f"GPU Total time (including transfers): $gpuTotalTimeMs%.3f ms")
    
    // --- GPU Bench (kernel only) ---
    // Warm up
    val d_a2 = CuArray.ones(n)
    val d_b2 = CuArray.ones(n)
    (d_a2 + d_b2).relu
    
    val startGpuKernel = System.nanoTime()
    val d_res2 = (d_a2 + d_b2).relu
    val endGpuKernel = System.nanoTime()
    val gpuKernelTimeMs = (endGpuKernel - startGpuKernel) / 1e6
    println(f"GPU Kernel time (approx): $gpuKernelTimeMs%.3f ms")
    
    // Verification
    var correct = true
    for (i <- 0 until 100) { // Check first 100
      if (Math.abs(h_res(i) - h_gpu_res(i)) > 1e-5) correct = false
    }
    println(s"Verification: ${if (correct) "PASSED" else "FAILED"}")
    
    println(f"Speedup (CPU / GPU Kernel): ${cpuTimeMs / gpuKernelTimeMs}%.2fx")
    println(f"Speedup (CPU / GPU Total): ${cpuTimeMs / gpuTotalTimeMs}%.2fx")
    
    backend.close()
