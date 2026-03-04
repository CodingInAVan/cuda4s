package com.cuda4s.jit.ir

import munit.FunSuite
import com.cuda4s.jit.ir.*
import com.cuda4s.jit.ir.DSL.*

class DSLSpec extends FunSuite {
  test("Implicit conversions and basic arithmetic") {
    val p: SVal = Param("x", Ty.F32)
    val v1: SVal = 10.s
    val v2: SVal = 5.0f.s
    
    val add = v1 + v2
    val sub = v1 - p
    val mul = p * v2
    val div = p / v1
    val lt = p < v1

    assertEquals(add.expr, Add(ConstI32(10), ConstF32(5.0f)))
    assertEquals(sub.expr, Sub(ConstI32(10), p.expr))
    assertEquals(mul.expr, Mul(p.expr, ConstF32(5.0f)))
    assertEquals(div.expr, Div(p.expr, ConstI32(10)))
    assertEquals(lt.expr, Lt(p.expr, ConstI32(10)))
  }

  test("Unsigned and explicit suffixes") {
    assertEquals(10.s.expr, ConstI32(10))
    assertEquals(10.u.expr, ConstU32(10))
    assertEquals(1.5f.s.expr, ConstF32(1.5f))
    assertEquals(2.5.s.expr, ConstF64(2.5))
  }

  test("Pointer operations") {
    val p: SPtr = Param("a", Ty.Ptr(Ty.F32))
    val i: SVal = threadIdx.x
    
    val load = p(i)
    assertEquals(load.expr, Load(p.expr, i.expr))
    
    val deref = p.unary_*
    assertEquals(deref.expr, Load(p.expr, ConstI32(0)))
    
    val offset = p + 5.s
    assertEquals(offset.expr, Add(p.expr, ConstI32(5)))

    val subOffset = p - 2.s
    assertEquals(subOffset.expr, Sub(p.expr, ConstI32(2)))
    
    val at = p.at(10.s)
    assertEquals(at.expr, Load(p.expr, ConstI32(10)))
  }

  test("Intrinsics") {
    assertEquals(threadIdx.x.expr, Intrinsic("threadIdx.x", Ty.I32))
    assertEquals(threadIdx.y.expr, Intrinsic("threadIdx.y", Ty.I32))
    assertEquals(threadIdx.z.expr, Intrinsic("threadIdx.z", Ty.I32))
    assertEquals(blockIdx.x.expr, Intrinsic("blockIdx.x", Ty.I32))
    assertEquals(blockIdx.y.expr, Intrinsic("blockIdx.y", Ty.I32))
    assertEquals(blockIdx.z.expr, Intrinsic("blockIdx.z", Ty.I32))
    assertEquals(blockDim.x.expr, Intrinsic("blockDim.x", Ty.I32))
    assertEquals(blockDim.y.expr, Intrinsic("blockDim.y", Ty.I32))
    assertEquals(blockDim.z.expr, Intrinsic("blockDim.z", Ty.I32))
  }

  test("Builder stmts") {
    val builder = new Builder()
    val p: SPtr = Param("a", Ty.Ptr(Ty.F32))
    val v: SVal = 1.0f.s
    
    p.update(0.s, v)(using builder)
    syncThreads()(using builder)
    
    val block = builder.buildBlock()
    assertEquals(block.stmts.length, 2)
    assert(block.stmts(0).isInstanceOf[Store])
    assert(block.stmts(1) == SyncThreadsStmt)
  }

  test("IfThen and nesting") {
    val builder = new Builder()
    ifThen(1.s < 2.s) {
      syncThreads()(using summon[Builder])
    }(using builder)
    
    val block = builder.buildBlock()
    assertEquals(block.stmts.length, 1)
    val ifStmt = block.stmts.head.asInstanceOf[IfThen]
    assertEquals(ifStmt.cond, Lt(ConstI32(1), ConstI32(2)))
    assertEquals(ifStmt.thenBody.stmts.head, SyncThreadsStmt)
  }

  test("Shared memory and device functions") {
    val params = List(Param("a", Ty.Ptr(Ty.F32)))
    val kDef = DSLCompiler.compileDSL("testK", params) {
      val sm = sharedMem("s", Ty.F32, 64)
      val df = DSLCompiler.deviceFunc("myFunc", List(Param("x", Ty.F32)), Ty.F32) {
        val x = SVal(Param("x", Ty.F32))
        (x * x).expr
      }
      sm(0) = call(df, 10.0f)
    }

    assertEquals(kDef.sharedMems.length, 1)
    assertEquals(kDef.sharedMems.head.name, "s")
    assertEquals(kDef.deviceFuncs.length, 1)
    assertEquals(kDef.deviceFuncs.head.name, "myFunc")
  }
}
