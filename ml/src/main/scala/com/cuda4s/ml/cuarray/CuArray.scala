package com.cuda4s.ml.cuarray

import com.cuda4s.jit.runtime.{CompiledKernel, GpuBackend, GpuBuffer}

import scala.collection.mutable
import com.cuda4s.jit.ir.*
import com.cuda4s.jit.ir.DSL.*
import com.cuda4s.jit.ir.DSLCompiler

class CuArray(val data: GpuBuffer, val shape: Seq[Int])(using backend: GpuBackend):
  val size: Int = shape.product

  def free(): Unit = backend.jni.free(data.dptr)

  // --- Element-wise Operations ---
  def +(other: CuArray): CuArray = elementWiseOp(other, CuArrayKernels.getAddKernel)
  def -(other: CuArray): CuArray = elementWiseOp(other, CuArrayKernels.getSubKernel)
  def *(other: CuArray): CuArray = elementWiseOp(other, CuArrayKernels.getMulKernel)
  def /(other: CuArray): CuArray = elementWiseOp(other, CuArrayKernels.getDivKernel)

  private def elementWiseOp(other: CuArray, getKernel: GpuBackend => CompiledKernel): CuArray =
    require(shape == other.shape, s"Shapes must match")
    val out = backend.allocateBuffer(size)
    val kernel = getKernel(backend)
    // Launch: (items=size, a, b, c, n=size)
    kernel.launch(size, data, other.data, out, size)
    new CuArray(out, shape)

  // --- Activation ---
  def relu: CuArray =
    val out = backend.allocateBuffer(size)
    val kernel = CuArrayKernels.getReluKernel(backend)
    kernel.launch(size, size, data, out)
    new CuArray(out, shape)

  // --- Matmul ---
  def matmul(other: CuArray): CuArray =
    require(shape.length == 2 && other.shape.length == 2, "2D only")
    require(shape(1) == other.shape(0), "Dim mismatch")
    val M = shape(0)
    val K = shape(1)
    val N = other.shape(1)
    val out = backend.allocateBuffer(M * N)
    val kernel = CuArrayKernels.getMatmulKernel(backend)

    // Launch: (items=total, A, B, C, M, K, N, total)
    val totalElements = M * N
    kernel.launch(totalElements, data, other.data, out, M, K, N, totalElements)
    new CuArray(out, Seq(M, N))

  // --- Reduction ---
  def sum: Float =
    val out = backend.allocateBuffer(1)
    val kernel = CuArrayKernels.getSumKernel(backend)
    kernel.blockSize = 1
    kernel.gridX = Some(1) // Force 1 block
    // Launch: (items=1, in, out, n=size)
    kernel.launch(1, data, out, size)

    val h_res = new Array[Float](1)
    out.copyTo(h_res)
    backend.jni.free(out.dptr)
    h_res(0)

  def mean: Float = sum / size

  def copyToHost(dest: Array[Float]): Unit = data.copyTo(dest)
  override def toString: String = s"CuArray(shape=$shape)"

object CuArray:
  def apply(values: Array[Float], shape: Int*)(using backend: GpuBackend): CuArray =
    val finalShape = if (shape.isEmpty) Seq(values.length) else shape.toSeq
    val buffer = backend.allocateBuffer(values.length)
    buffer.copyFrom(values)
    new CuArray(buffer, finalShape)

  def zeros(shape: Int*)(using backend: GpuBackend): CuArray = fill(0.0f, shape*)
  def ones(shape: Int*)(using backend: GpuBackend): CuArray = fill(1.0f, shape*)

  def fill(value: Float, shape: Int*)(using backend: GpuBackend): CuArray =
    val size = shape.product
    val buffer = backend.allocateBuffer(size)
    val kernel = CuArrayKernels.getFillKernel(backend)
    // Launch: (items=size, out, v, n)
    kernel.launch(size, buffer, value, size)
    new CuArray(buffer, shape.toSeq)


object CuArrayKernels:

  // We use CompiledKernel (not Kernel) everywhere for consistency
  private val cache = mutable.WeakHashMap[GpuBackend, mutable.Map[String, CompiledKernel]]()

  // Updated to accept a DSL body: (Builder ?=> Unit)
  private def getOrCompile(name: String, paramDefs: List[Param], backend: GpuBackend)(body: Builder ?=> Unit): CompiledKernel =
    val backendCache = cache.getOrElseUpdate(backend, mutable.Map.empty)
    backendCache.getOrElseUpdate(name, {
      // Use compileDSL explicitly for Builder/SVal based logic
      val defn = DSLCompiler.compileDSL(name, paramDefs)(body)
      backend.compile(defn)
    })

  // --- Kernels ---

  // Order: 0->a, 1->b, 2->c, 3->n
  private val binaryOpParams = List(Param("a", Ty.Ptr(Ty.F32)), Param("b", Ty.Ptr(Ty.F32)), Param("c", Ty.Ptr(Ty.F32)), Param("n", Ty.I32))

  def getAddKernel(b: GpuBackend): CompiledKernel = getOrCompile("add", binaryOpParams, b) {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    val n = binaryOpParams(3) // Access param list directly
    val a = binaryOpParams(0)
    val b = binaryOpParams(1)
    val c = binaryOpParams(2)
    ifThen(i < n) {
      val A = SPtr(a)
      val B = SPtr(b)
      val C = SPtr(c)
      C(i) = A(i) + B(i)
    }
  }

  def getSubKernel(b: GpuBackend): CompiledKernel = getOrCompile("sub", binaryOpParams, b) {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    val n = binaryOpParams(3)
    ifThen(i < n) {
      val A = SPtr(binaryOpParams(0))
      val B = SPtr(binaryOpParams(1))
      val C = SPtr(binaryOpParams(2))
      C(i) = A(i) - B(i)
    }
  }

  def getMulKernel(b: GpuBackend): CompiledKernel = getOrCompile("mul", binaryOpParams, b) {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    val n = binaryOpParams(3)
    ifThen(i < n) {
      val A = SPtr(binaryOpParams(0))
      val B = SPtr(binaryOpParams(1))
      val C = SPtr(binaryOpParams(2))
      C(i) = A(i) * B(i)
    }
  }

  def getDivKernel(b: GpuBackend): CompiledKernel = getOrCompile("div", binaryOpParams, b) {
    val i = blockIdx.x * blockDim.x + threadIdx.x
    val n = binaryOpParams(3)
    ifThen(i < n) {
      val A = SPtr(binaryOpParams(0))
      val B = SPtr(binaryOpParams(1))
      val C = SPtr(binaryOpParams(2))
      C(i) = A(i) / B(i)
    }
  }

  def getReluKernel(b: GpuBackend): CompiledKernel =
    val p = List(Param("n", Ty.I32), Param("in", Ty.Ptr(Ty.F32)), Param("out", Ty.Ptr(Ty.F32)))
    getOrCompile("relu", p, b) {
      val i = blockIdx.x * blockDim.x + threadIdx.x
      val n = p(0)
      ifThen(i < n) {
        val in = SPtr(p(1))
        val out = SPtr(p(2))
        val v = in(i)
        // DSL conditional assignment
        ifThenElse(v > 0.0f) {
          out(i) = v
        } {
          out(i) = 0.0f.s // .s converts float to SVal
        }
      }
    }

  def getFillKernel(b: GpuBackend): CompiledKernel =
    val p = List(Param("out", Ty.Ptr(Ty.F32)), Param("v", Ty.F32), Param("n", Ty.I32))
    getOrCompile("fill", p, b) {
      val i = blockIdx.x * blockDim.x + threadIdx.x
      val n = p(2)
      ifThen(i < n) {
        val out = SPtr(p(0))
        val v = SVal(p(1))
        out(i) = v
      }
    }

  def getMatmulKernel(b: GpuBackend): CompiledKernel =
    val p = List(Param("A", Ty.Ptr(Ty.F32)), Param("B", Ty.Ptr(Ty.F32)), Param("C", Ty.Ptr(Ty.F32)), Param("M", Ty.I32), Param("K", Ty.I32), Param("N", Ty.I32), Param("size", Ty.I32))
    getOrCompile("matmul", p, b) {
      val idx = blockIdx.x * blockDim.x + threadIdx.x
      val total = p(6)

      ifThen(idx < total) {
        val N = p(5)
        val K = p(4)
        val row = idx / N
        val col = idx % N

        val sum = varDef("sum", Ty.F32, 0.0f.s)
        forRange(0.s, K) { k =>
          val A = SPtr(p(0))
          val B = SPtr(p(1))
          assign(sum, sum + (A(row * K + k) * B(k * N + col)))
        }
        val C = SPtr(p(2))
        C(idx) = sum
      }
    }

  def getSumKernel(b: GpuBackend): CompiledKernel =
    val p = List(Param("in", Ty.Ptr(Ty.F32)), Param("out", Ty.Ptr(Ty.F32)), Param("n", Ty.I32))
    getOrCompile("sum", p, b) {
      val i = blockIdx.x * blockDim.x + threadIdx.x
      ifThen(i === 0.s) {
        val n = p(2)
        val in = SPtr(p(0))

        val s = varDef("s", Ty.F32, 0.0f.s)
        forRange(0.s, n) { j =>
          assign(s, s + in(j))
        }
        val out = SPtr(p(1))
        out(0.s) = s
      }
    }