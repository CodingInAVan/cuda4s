package com.cuda4s.jit.examples

import com.cuda4s.jit.syntax.CudaSyntax.*
import com.cuda4s.jit.runtime.GpuBackend

object CPointerStyleExample:

  val cStyleSrc = cuda"""
    |extern "C" __global__
    |void cStyleKernel(const float* a, const float* b, float* c, int n) {
    |  int i = blockIdx.x * blockDim.x + threadIdx.x;
    |  if (i < n) {
    |    *(c + i) = *(a + i) + *(b + i);
    |  }
    |}
    """

  val scalarPointerSrc = cuda"""
    |extern "C" __global__
    |void scalarPointerKernel(float* ptr, float valIn) {
    |  *ptr = *ptr + valIn;
    |}
    """

  def main(args: Array[String]): Unit =
    println("CUDA source with C-style pointer arithmetic:")
    println("--------------------------------------------------")
    println(cStyleSrc)
    println("--------------------------------------------------")

    val backend = new GpuBackend()
    try
      backend.compileRaw("cStyleKernel", cStyleSrc)
      println("cStyleKernel compiled successfully!")
      backend.compileRaw("scalarPointerKernel", scalarPointerSrc)
      println("scalarPointerKernel compiled successfully!")
    finally
      backend.close()
