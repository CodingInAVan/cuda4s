package com.cuda4s

final class CudaJitJNI {
  @native def init(kernelSrc: String, kernelName: String): Long
  @native def close(handle: Long): Unit

  // Generic CUDA functions
  @native def alloc(bytes: Long): Long
  @native def allocMany(bytes: Array[Long]): Array[Long]
  @native def free(ptr: Long): Unit
  @native def freeMany(ptrs: Array[Long]): Unit
  @native def copyHtoD(ptr: Long, src: Array[Float], bytes: Long): Unit
  @native def copyDtoH(dest: Array[Float], ptr: Long, bytes: Long): Unit
  @native def copyHtoDInts(ptr: Long, src: Array[Int], bytes: Long): Unit
  @native def copyDtoHInts(dest: Array[Int], ptr: Long, bytes: Long): Unit
  @native def copyHtoDDoubles(ptr: Long, src: Array[Double], bytes: Long): Unit
  @native def copyDtoHDoubles(dest: Array[Double], ptr: Long, bytes: Long): Unit
  @native def copyHtoDLongs(ptr: Long, src: Array[Long], bytes: Long): Unit
  @native def copyDtoHLongs(dest: Array[Long], ptr: Long, bytes: Long): Unit
  @native def launch(handle: Long, gridX: Int, blockX: Int, args: Array[Long]): Unit
}

object CudaJitJNI {
  System.loadLibrary("cudajitjni")
  val instance = new CudaJitJNI()
}

