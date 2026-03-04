package com.cuda4s.jit.runtime

import com.cuda4s.CudaJitJNI

class GpuBuffer(val sizeBytes: Long, val dptr: Long, jni: CudaJitJNI, private var owned: Boolean = true) {
  private var freed = false

  def copyFrom(src: Array[Float]): Unit = {
    require(!freed, "Buffer already freed")
    jni.copyHtoD(dptr, src, sizeBytes)
  }

  def copyTo(dest: Array[Float]): Unit = {
    require(!freed, "Buffer already freed")
    jni.copyDtoH(dest, dptr, sizeBytes)
  }

  def copyFrom(src: Array[Int]): Unit = {
    require(!freed, "Buffer already freed")
    jni.copyHtoDInts(dptr, src, sizeBytes)
  }

  def copyTo(dest: Array[Int]): Unit = {
    require(!freed, "Buffer already freed")
    jni.copyDtoHInts(dest, dptr, sizeBytes)
  }

  def copyFrom(src: Array[Double]): Unit = {
    require(!freed, "Buffer already freed")
    jni.copyHtoDDoubles(dptr, src, sizeBytes)
  }

  def copyTo(dest: Array[Double]): Unit = {
    require(!freed, "Buffer already freed")
    jni.copyDtoHDoubles(dest, dptr, sizeBytes)
  }

  def copyFrom(src: Array[Long]): Unit = {
    require(!freed, "Buffer already freed")
    jni.copyHtoDLongs(dptr, src, sizeBytes)
  }

  def copyTo(dest: Array[Long]): Unit = {
    require(!freed, "Buffer already freed")
    jni.copyDtoHLongs(dest, dptr, sizeBytes)
  }

  def free(): Unit = {
    if (!freed && owned) {
      jni.free(dptr)
      freed = true
    }
  }

  override def finalize(): Unit = {
    if (!freed && owned) free()
  }
}
