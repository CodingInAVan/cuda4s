package com.cuda4s.jit.examples

import com.cuda4s.jit.syntax.CudaSyntax.*

object VecAddExample:

  val vecAddSrc = cuda"""
    |extern "C" __global__
    |void vecAdd(const float* a, const float* b, float* c, int n) {
    |  int i = blockIdx.x * blockDim.x + threadIdx.x;
    |  if (i < n) c[i] = a[i] + b[i];
    |}
    """

  def main(args: Array[String]): Unit =
    println("CUDA source via cuda\"...\" interpolator:")
    println("-----------------------")
    println(vecAddSrc)
    println("-----------------------")

    import com.cuda4s.jit.runtime.GpuBackend
    val n = 1024
    val backend = new GpuBackend()
    try
      val kernel = backend.compileRaw("vecAdd", vecAddSrc)
      println("Kernel compiled successfully via NVRTC!")
    finally
      backend.close()
