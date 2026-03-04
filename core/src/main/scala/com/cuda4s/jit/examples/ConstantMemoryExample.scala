package com.cuda4s.jit.examples

import com.cuda4s.jit.syntax.CudaSyntax.*
import com.cuda4s.jit.runtime.GpuBackend

object ConstantMemoryExample:

  val constantKernelSrc = cuda"""
    |__constant__ float table[2];
    |
    |extern "C" __global__
    |void constantKernel(const float* a, float* c, int n) {
    |  int i = blockIdx.x * blockDim.x + threadIdx.x;
    |  if (i < n) {
    |    c[i] = a[i] * table[0] + table[1];
    |  }
    |}
    """

  def main(args: Array[String]): Unit =
    println("CUDA source with __constant__ memory:")
    println("--------------------------------------------------")
    println(constantKernelSrc)
    println("--------------------------------------------------")

    val backend = new GpuBackend()
    try
      val kernel = backend.compileRaw("constantKernel", constantKernelSrc)
      println("Kernel compiled successfully!")
    finally
      backend.close()
