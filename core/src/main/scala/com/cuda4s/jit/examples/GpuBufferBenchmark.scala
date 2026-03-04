package com.cuda4s.jit.examples

import com.cuda4s.jit.syntax.CudaSyntax.*
import com.cuda4s.jit.runtime.GpuBackend
import com.cuda4s.jit.utils.BenchmarkUtils.timeMs

object GpuBufferBenchmark:

  val vecAddSrc = cuda"""
    |extern "C" __global__
    |void vecAdd(const float* a, const float* b, float* c, int n) {
    |  int i = blockIdx.x * blockDim.x + threadIdx.x;
    |  if (i < n) c[i] = a[i] + b[i];
    |}
    """

  def main(args: Array[String]): Unit =
    val n = 1 << 20
    val a = Array.tabulate(n)(i => i.toFloat)
    val b = Array.tabulate(n)(i => (2 * i).toFloat)
    val outGpu = new Array[Float](n)

    val backend = new GpuBackend()

    try
      val kernel = backend.compileRaw("vecAdd", vecAddSrc)
      kernel.blockSize = 1024

      val bufA = backend.allocateBuffer(n)
      val bufB = backend.allocateBuffer(n)
      val bufC = backend.allocateBuffer(n)
      bufA.copyFrom(a)
      bufB.copyFrom(b)

      println(s"Benchmarking n=$n")

      timeMs("GPU run (GpuBuffer - manual copy)", warmup = 10, iters = 100) {
        bufA.copyFrom(a)
        bufB.copyFrom(b)
        kernel.launch(n, bufA, bufB, bufC)
        bufC.copyTo(outGpu)
        0.0
      }

      timeMs("GPU launch only (Resident data)", warmup = 10, iters = 100) {
        kernel.launch(n, bufA, bufB, bufC)
        0.0
      }

      bufA.free()
      bufB.free()
      bufC.free()

    finally
      backend.close()
