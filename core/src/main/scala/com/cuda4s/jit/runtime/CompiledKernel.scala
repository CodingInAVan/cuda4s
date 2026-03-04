package com.cuda4s.jit.runtime

import com.cuda4s.CudaJitJNI

class CompiledKernel(val handle: Long, jni: CudaJitJNI):
  var blockSize: Int = 256
  // Optional: Manually override grid size. If None, calculated from 'items'
  var gridX: Option[Int] = None

  /**
   * Launch the kernel.
   *
   * @param items The total number of threads required (global work size). Used to calculate grid dim.
   * @param args  The arguments to pass to the CUDA kernel (GpuBuffer, Int, Float, etc.)
   */
  def launch(items: Int, args: Any*): Unit =
    val gx = gridX.getOrElse((items + blockSize - 1) / blockSize)
    // Fix: Pass 'args' directly. launchRaw expects Seq[Any], not varargs.
    launchRaw(gx, blockSize, args)

  /**
   * Launch with a manual grid configuration.
   */
  def launchManual(gridSize: Int, blockSize: Int, args: Any*): Unit =
    // Fix: Pass 'args' directly.
    launchRaw(gridSize, blockSize, args)

  private def launchRaw(gx: Int, bs: Int, args: Seq[Any]): Unit =
    val longArgs = args.map {
      case b: GpuBuffer => b.dptr
      case i: Int => i.toLong
      case f: Float => java.lang.Float.floatToRawIntBits(f).toLong
      case d: Double => java.lang.Double.doubleToRawLongBits(d)
      case l: Long => l
      case other => throw new IllegalArgumentException(s"Unsupported kernel arg type: ${other.getClass}")
    }.toArray
    jni.launch(handle, gx, bs, longArgs)