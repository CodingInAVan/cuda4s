package com.cuda4s.jit.examples

import com.cuda4s.jit.syntax.CudaSyntax.*
import com.cuda4s.jit.runtime.GpuBackend

object BenchmarkExample:

  val vecAddSrc = cuda"""
    |extern "C" __global__
    |void vecAdd(const float* a, const float* b, float* c, int n) {
    |  int i = blockIdx.x * blockDim.x + threadIdx.x;
    |  if (i < n) c[i] = a[i] + b[i];
    |}
    """

  def vecAddCpu(a: Array[Float], b: Array[Float], out: Array[Float]): Unit =
    val n = a.length
    var i = 0
    while (i < n) do
      out(i) = a(i) + b(i)
      i += 1

  def main(args: Array[String]): Unit =
    val n = 1 << 24
    val size = n * 4L
    println(f"Vector Addition Benchmark (n = $n elements)")
    println(f"Data size: ${size.toDouble / (1024 * 1024)}%.2f MB per vector")

    val a = Array.tabulate(n)(i => i.toFloat)
    val b = Array.tabulate(n)(i => (i * 2).toFloat)
    val outCpu = new Array[Float](n)
    val outGpu = new Array[Float](n)

    val backend = new GpuBackend()
    backend.setBlockSize(256)

    try
      val startCpu = System.nanoTime()
      vecAddCpu(a, b, outCpu)
      val endCpu = System.nanoTime()
      val durationCpuMs = (endCpu - startCpu) / 1e6
      println(f"CPU time: $durationCpuMs%.3f ms")

      val kernel = backend.compileRaw("vecAdd", vecAddSrc)
      val bufA = backend.allocateBuffer(n)
      val bufB = backend.allocateBuffer(n)
      val bufC = backend.allocateBuffer(n)

      try
        val startGpuTotal = System.nanoTime()
        bufA.copyFrom(a)
        bufB.copyFrom(b)

        val startKernel = System.nanoTime()
        kernel.launch(n, bufA, bufB, bufC)
        val endKernel = System.nanoTime()

        bufC.copyTo(outGpu)
        val endGpuTotal = System.nanoTime()

        val kernelMs = (endKernel - startKernel) / 1e6
        val totalGpuMs = (endGpuTotal - startGpuTotal) / 1e6

        println(f"GPU Kernel time: $kernelMs%.3f ms")
        println(f"GPU Total time (including H2D and D2H): $totalGpuMs%.3f ms")
        println(f"Speedup (CPU / GPU Kernel): ${durationCpuMs / kernelMs}%.2fx")
        println(f"Speedup (CPU / GPU Total): ${durationCpuMs / totalGpuMs}%.2fx")

        var correct = true
        var i = 0
        while (i < n && correct) {
          if (math.abs(outCpu(i) - outGpu(i)) > 1e-5) correct = false
          i += 1
        }
        println(s"Verification: ${if (correct) "PASSED" else "FAILED"}")

      finally
        bufA.free()
        bufB.free()
        bufC.free()

    finally
      backend.close()
