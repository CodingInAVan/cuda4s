package com.cuda4s.jit.runtime

import com.cuda4s.jit.codegen.CudaCodegen
import com.cuda4s.jit.ir.KernelDef
import com.cuda4s.CudaJitJNI

class GpuBackend:
  val jni = CudaJitJNI.instance
  private val cache = new KernelCache((src, name) => jni.init(src, name))
  private var defaultBlockSize = 256

  /**
   * Allocates a buffer for 'size' Float elements (size * 4 bytes).
   */
  def allocateBuffer(size: Int): GpuBuffer =
    val sizeBytes = size.toLong * 4
    val dptr = jni.alloc(sizeBytes)
    new GpuBuffer(sizeBytes, dptr, jni)

  def allocateBytes(bytes: Long): GpuBuffer =
    val dptr = jni.alloc(bytes)
    new GpuBuffer(bytes, dptr, jni)

  def allocateBuffers(sizes: Seq[Int]): Seq[GpuBuffer] =
    val bytes = sizes.map(_.toLong * 4).toArray
    val ptrs = jni.allocMany(bytes)
    ptrs.zip(bytes).map { case (ptr, sz) => new GpuBuffer(sz, ptr, jni) }.toSeq

  def setBlockSize(bs: Int): Unit = { defaultBlockSize = bs }

  def compile(kernelDef: KernelDef): CompiledKernel =
    val cudaSrc = CudaCodegen.generate(kernelDef)
    val handle = cache.getOrCompile(kernelDef, cudaSrc)
    val compiledKernel = new CompiledKernel(handle, jni)
    compiledKernel.blockSize = defaultBlockSize
    compiledKernel

  def compileRaw(name: String, source: String): CompiledKernel =
    val handle = cache.getOrCompile(name, source)
    val compiledKernel = new CompiledKernel(handle, jni)
    compiledKernel.blockSize = defaultBlockSize
    compiledKernel

  def close(): Unit =
    cache.closeAll(jni.close)
