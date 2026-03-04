package com.cuda4s.ml.examples

import com.cuda4s.jit.runtime.GpuBackend
import com.cuda4s.ml.cuarray.CuArray

object MatMulExample:
  def main(args: Array[String]): Unit =
    val M = 512
    val K = 512
    val N = 512
    
    println(s"Benchmark: Matrix Multiplication (MxK * KxN)")
    println(s"Dimensions: $M x $K * $K x $N")
    
    given backend: GpuBackend = new GpuBackend()
    
    // Initialize data
    val h_a = Array.fill(M * K)(1.0f)
    val h_b = Array.fill(K * N)(2.0f)
    
    // --- CPU Bench ---
    val startCpu = System.nanoTime()
    val h_res = new Array[Float](M * N)
    for (m <- 0 until M) {
      for (n <- 0 until N) {
        var sum = 0.0f
        for (k <- 0 until K) {
          sum += h_a(m * K + k) * h_b(k * N + n)
        }
        h_res(m * N + n) = sum
      }
    }
    val endCpu = System.nanoTime()
    val cpuTimeMs = (endCpu - startCpu) / 1e6
    println(f"CPU time: $cpuTimeMs%.3f ms")
    
    // --- GPU Bench (including transfers) ---
    val startGpuTotal = System.nanoTime()
    val d_a = CuArray(h_a, M, K)
    val d_b = CuArray(h_b, K, N)
    
    val d_res = d_a.matmul(d_b)
    
    val h_gpu_res = new Array[Float](M * N)
    d_res.copyToHost(h_gpu_res)
    val endGpuTotal = System.nanoTime()
    val gpuTotalTimeMs = (endGpuTotal - startGpuTotal) / 1e6
    println(f"GPU Total time (including transfers): $gpuTotalTimeMs%.3f ms")
    
    // --- GPU Bench (kernel only) ---
    // Warm up
    val d_a2 = CuArray.fill(1.0f, M, K)
    val d_b2 = CuArray.fill(2.0f, K, N)
    d_a2.matmul(d_b2)
    
    val startGpuKernel = System.nanoTime()
    val d_res2 = d_a2.matmul(d_b2)
    val endGpuKernel = System.nanoTime()
    val gpuKernelTimeMs = (endGpuKernel - startGpuKernel) / 1e6
    println(f"GPU Kernel time (approx): $gpuKernelTimeMs%.3f ms")
    
    // Verification
    var correct = true
    for (i <- 0 until M * N) {
      if (Math.abs(h_res(i) - h_gpu_res(i)) > 1e-3) {
        if (correct) {
          println(s"Error at index $i: CPU=${h_res(i)}, GPU=${h_gpu_res(i)}")
        }
        correct = false
      }
    }
    println(s"Verification: ${if (correct) "PASSED" else "FAILED"}")
    
    val flops = 2.0 * M * K * N
    val gflopsCpu = (flops / (cpuTimeMs / 1000.0)) / 1e9
    val gflopsGpu = (flops / (gpuKernelTimeMs / 1000.0)) / 1e9
    
    println(f"CPU Performance: $gflopsCpu%.3f GFLOPS")
    println(f"GPU Performance: $gflopsGpu%.3f GFLOPS")
    println(f"Speedup (CPU / GPU Kernel): ${cpuTimeMs / gpuKernelTimeMs}%.2fx")
    
    backend.close()
