package com.cuda4s.jit.ir

import com.cuda4s.jit.codegen.CudaCodegen
import munit.FunSuite
import com.cuda4s.jit.ir.*

class CodegenSpec extends FunSuite {
  test("Basic CUDA code generation") {
    val params = List(
      Param("a", Ty.Ptr(Ty.F32)),
      Param("n", Ty.I32)
    )
    val body = Block(List(
      Store(Param("a", Ty.Ptr(Ty.F32)), Intrinsic("threadIdx.x", Ty.I32), ConstF32(1.0f))
    ))
    val kernel = KernelDef("simple", params, body)
    
    val code = CudaCodegen.generate(kernel)
    assert(code.contains("void simple(float* a, int n)"))
    assert(code.contains("a[threadIdx.x] = 1.0f;"))
  }

  test("CSE and Common expressions") {
    val p = Param("p", Ty.F32)
    val shared = Add(p, ConstF32(1.0f))
    val body = Block(List(
      Store(Param("out", Ty.Ptr(Ty.F32)), ConstI32(0), Add(shared, shared))
    ))
    val kernel = KernelDef("cse_test", List(p, Param("out", Ty.Ptr(Ty.F32))), body)
    
    val code = CudaCodegen.generate(kernel)
    // Check that v0 is created for (p + 1.0f)
    assert(code.contains("float v0 = (p + 1.0f);"))
    assert(code.contains("out[0] = (v0 + v0);"))
  }

  test("Constant parameters and shared memory") {
    val params = List(Param("c", Ty.I32, isConstant = true))
    val sm = List(SharedMem("cache", Ty.F32, 256))
    val kernel = KernelDef("const_sm", params, Block(Nil), sharedMems = sm)
    
    val code = CudaCodegen.generate(kernel)
    assert(code.contains("__constant__ int c"))
    assert(code.contains("__shared__ float cache[256];"))
  }

  test("Device functions") {
    val dfParams = List(Param("x", Ty.F32))
    val df = DeviceFuncDef("sq", dfParams, Block(Nil), Mul(Param("x", Ty.F32), Param("x", Ty.F32)), Ty.F32)
    val body = Block(List(
      Store(Param("out", Ty.Ptr(Ty.F32)), ConstI32(0), Call("sq", List(ConstF32(2.0f))))
    ))
    val kernel = KernelDef("use_df", List(Param("out", Ty.Ptr(Ty.F32))), body, deviceFuncs = List(df))
    
    val code = CudaCodegen.generate(kernel)
    assert(code.contains("__device__ float sq(float x);")) // forward decl
    assert(code.contains("__device__ float sq(float x) {")) // definition
    assert(code.contains("return (x * x);"))
    assert(code.contains("out[0] = sq(2.0f);"))
  }

  test("All expression types in codegen") {
    val p = Param("p", Ty.I32)
    val u = Param("u", Ty.U32)
    val f = Param("f", Ty.F32)
    
    val exprs = List(
      Add(p, ConstI32(1)),
      Sub(u, ConstU32(1)),
      Mul(f, ConstF32(1.1f)),
      Div(p, ConstI32(2)),
      Lt(p, ConstI32(10)),
      Call("abs", List(p)),
      Intrinsic("blockIdx.x", Ty.I32),
      Load(Param("ptr", Ty.Ptr(Ty.I32)), ConstI32(0)),
      SyncThreads
    )
    
    val body = Block(exprs.map(e => Store(Param("out", Ty.Ptr(Ty.I32)), ConstI32(0), e)))
    val kernel = KernelDef("all_exprs", List(p, u, f, Param("out", Ty.Ptr(Ty.I32))), body)
    
    val code = CudaCodegen.generate(kernel)
    assert(code.contains("1u"))
    assert(code.contains("1.1f"))
    assert(code.contains("__syncthreads()"))
    assert(code.contains("abs(p)"))
    assert(code.contains("<"))
    assert(code.contains("/"))
  }

  test("Type inference for different expressions") {
    // This indirectly tests inferType through CSE variable generation
    val f = Param("f", Ty.F32)
    val repeatedF = Add(f, f)
    
    val i = Param("i", Ty.I32)
    val repeatedI = Add(i, i)
    
    val u = Param("u", Ty.U32)
    val repeatedU = Add(u, u)
    
    val body = Block(List(
      Store(Param("outF", Ty.Ptr(Ty.F32)), ConstI32(0), Add(repeatedF, repeatedF)),
      Store(Param("outI", Ty.Ptr(Ty.I32)), ConstI32(0), Add(repeatedI, repeatedI)),
      Store(Param("outU", Ty.Ptr(Ty.U32)), ConstI32(0), Add(repeatedU, repeatedU))
    ))
    
    val kernel = KernelDef("type_inf", List(f, i, u, Param("outF", Ty.Ptr(Ty.F32)), Param("outI", Ty.Ptr(Ty.I32)), Param("outU", Ty.Ptr(Ty.U32))), body)
    val code = CudaCodegen.generate(kernel)
    
    assert(code.contains("(f + f)"))
    assert(code.contains("(i + i)"))
    assert(code.contains("(u + u)"))
  }

  test("CSE Depth sorting") {
    val x = Param("x", Ty.F32)
    val inner = Add(x, ConstF32(1.0f))
    val outer = Mul(inner, inner)
    val body = Block(List(
      Store(Param("out", Ty.Ptr(Ty.F32)), ConstI32(0), Add(outer, outer))
    ))
    
    val kernel = KernelDef("cse_depth", List(x, Param("out", Ty.Ptr(Ty.F32))), body)
    val code = CudaCodegen.generate(kernel)
    
    // Check that nested expressions are emitted in correct order
    // v0 should be (x + 1.0f) and v1 should be (v0 * v0)
    assert(code.contains("float v0 = (x + 1.0f);"))
    assert(code.contains("float v1 = (v0 * v0);"))
    assert(code.indexOf("v0 =") < code.indexOf("v1 ="))
  }

  test("IfThenElse code generation") {
    val params = List(Param("a", Ty.F32), Param("out", Ty.Ptr(Ty.F32)))
    val body = Block(List(
      IfThenElse(Lt(Param("a", Ty.F32), ConstF32(0.0f)), 
        Block(List(Store(Param("out", Ty.Ptr(Ty.F32)), ConstI32(0), ConstF32(0.0f)))),
        Block(List(Store(Param("out", Ty.Ptr(Ty.F32)), ConstI32(0), Param("a", Ty.F32))))
      )
    ))
    val kernel = KernelDef("ifelse", params, body)
    val code = CudaCodegen.generate(kernel)
    assert(code.contains("if ((a < 0.0f)) {"))
    assert(code.contains("} else {"))
  }

  test("Comparison operators generation") {
    val params = List(Param("a", Ty.I32), Param("b", Ty.I32))
    val ops = List(
      Gt(Param("a", Ty.I32), Param("b", Ty.I32)),
      Le(Param("a", Ty.I32), Param("b", Ty.I32)),
      Ge(Param("a", Ty.I32), Param("b", Ty.I32)),
      Eq(Param("a", Ty.I32), Param("b", Ty.I32)),
      Ne(Param("a", Ty.I32), Param("b", Ty.I32))
    )
    val body = Block(ops.map(e => Store(Param("out", Ty.Ptr(Ty.I32)), ConstI32(0), e)))
    val kernel = KernelDef("comparisons", params :+ Param("out", Ty.Ptr(Ty.I32)), body)
    val code = CudaCodegen.generate(kernel)
    assert(code.contains(">"))
    assert(code.contains("<="))
    assert(code.contains(">="))
    assert(code.contains("=="))
    assert(code.contains("!="))
  }
}
