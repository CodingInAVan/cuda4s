package com.cuda4s.jit.runtime

import com.cuda4s.jit.ir.KernelDef
import scala.collection.mutable

class KernelCache(compileFn: (String, String) => Long) {
  private val cache = mutable.Map[String, Long]()

  def getOrCompile(kernelDef: KernelDef, cudaSrc: String): Long = {
    val key = s"${kernelDef.name}#${cudaSrc.hashCode}"
    cache.getOrElseUpdate(key, {
      compileFn(cudaSrc, kernelDef.name)
    })
  }

  def getOrCompile(name: String, cudaSrc: String): Long = {
    val key = s"$name#${cudaSrc.hashCode}"
    cache.getOrElseUpdate(key, {
      compileFn(cudaSrc, name)
    })
  }

  def closeAll(closeFn: Long => Unit): Unit = {
    cache.values.foreach(closeFn)
    cache.clear()
  }
}
