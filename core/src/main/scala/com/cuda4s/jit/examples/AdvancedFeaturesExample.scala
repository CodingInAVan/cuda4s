package com.cuda4s.jit.examples

import com.cuda4s.jit.syntax.CudaSyntax.*
import com.cuda4s.jit.runtime.GpuBackend

object AdvancedFeaturesExample:

  val advancedSrc = cuda"""
    |__device__ float square(float x) { return x * x; }
    |__device__ float cube(float x) { return square(x) * x; }
    |
    |extern "C" __global__
    |void advancedKernel(const float* a, const float* b, float* c, int n) {
    |  __shared__ float cache[256];
    |  int i = blockIdx.x * blockDim.x + threadIdx.x;
    |  int tid = threadIdx.x;
    |  if (i < n) {
    |    cache[tid] = a[i];
    |    __syncthreads();
    |    float cubed = cube(cache[tid]);
    |    c[i] = cubed + b[i];
    |  }
    |}
    """

  def main(args: Array[String]): Unit =
    println("Advanced CUDA kernel with device functions and shared memory:")
    println("--------------------------------------------------------------")
    println(advancedSrc)
    println("--------------------------------------------------------------")

    val backend = new GpuBackend()
    try
      val kernel = backend.compileRaw("advancedKernel", advancedSrc)
      println("Kernel compiled successfully via NVRTC!")
    finally
      backend.close()
