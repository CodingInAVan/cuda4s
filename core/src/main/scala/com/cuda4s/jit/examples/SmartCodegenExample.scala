package com.cuda4s.jit.examples

import com.cuda4s.jit.syntax.CudaSyntax.*
import com.cuda4s.jit.runtime.GpuBackend

object SmartCodegenExample:

  val smartKernelSrc = cuda"""
    |extern "C" __global__
    |void smartKernel(const float* a, const float* b, float* c, int n) {
    |  int i = blockIdx.x * blockDim.x + threadIdx.x;
    |  if (i < n) {
    |    float sum = a[i] + b[i];
    |    c[i] = sum * sum + sum;
    |  }
    |}
    """

  def main(args: Array[String]): Unit =
    println("CUDA source with manual CSE (sum reuse):")
    println("--------------------------------------------------")
    println(smartKernelSrc)
    println("--------------------------------------------------")

    val backend = new GpuBackend()
    try
      val kernel = backend.compileRaw("smartKernel", smartKernelSrc)
      println("Kernel compiled successfully!")
    finally
      backend.close()
