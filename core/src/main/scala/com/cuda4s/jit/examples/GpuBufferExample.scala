package com.cuda4s.jit.examples

import com.cuda4s.jit.syntax.CudaSyntax.*
import com.cuda4s.jit.runtime.GpuBackend

object GpuBufferExample:

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
      println(s"Allocating buffers for n=$n...")
      val bufA = backend.allocateBuffer(n)
      val bufB = backend.allocateBuffer(n)
      val bufC = backend.allocateBuffer(n)

      println("Copying data to GPU...")
      bufA.copyFrom(a)
      bufB.copyFrom(b)

      val kernel = backend.compileRaw("vecAdd", vecAddSrc)
      kernel.blockSize = 1024

      println("Launching kernel using GpuBuffer...")
      kernel.launch(n, bufA, bufB, bufC)

      println("Copying results back from GPU...")
      bufC.copyTo(outGpu)

      val ok = (0 until 100).forall(i => math.abs(outGpu(i) - (a(i) + b(i))) < 1e-4)
      println(s"Correctness: $ok")
      if ok then
        println(s"Sample results: outGpu(0)=${outGpu(0)}, outGpu(${n / 2})=${outGpu(n / 2)}")

      bufA.free()
      bufB.free()
      bufC.free()

    finally
      backend.close()
