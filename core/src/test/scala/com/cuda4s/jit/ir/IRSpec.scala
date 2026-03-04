package com.cuda4s.jit.ir

import munit.FunSuite

class IRSpec extends FunSuite {
  test("Ty equality") {
    assertEquals(Ty.I32, Ty.I32)
    assertEquals(Ty.U32, Ty.U32)
    assertEquals(Ty.F32, Ty.F32)
    assertEquals(Ty.Ptr(Ty.I32), Ty.Ptr(Ty.I32))
    assertNotEquals(Ty.I32, Ty.F32)
    assertNotEquals(Ty.Ptr(Ty.I32), Ty.Ptr(Ty.F32))
  }

  test("Expr structures") {
    val p1 = Param("p1", Ty.I32)
    val p2 = Param("p2", Ty.I32, isConstant = true)
    val add = Add(p1, ConstI32(10))
    val sub = Sub(add, ConstU32(5))
    val mul = Mul(sub, ConstF32(2.5f))
    val div = Div(mul, p2)
    val lt = Lt(div, ConstI32(0))
    val call = Call("func", List(p1, p2))
    val intrinsic = Intrinsic("threadIdx.x", Ty.I32)
    val load = Load(Param("ptr", Ty.Ptr(Ty.F32)), ConstI32(0))

    assertEquals(p1.name, "p1")
    assertEquals(p2.isConstant, true)
    assertEquals(add.lhs, p1)
    assertEquals(sub.rhs, ConstU32(5))
    assertEquals(mul.rhs, ConstF32(2.5f))
    assertEquals(div.rhs, p2)
    assertEquals(lt.lhs, div)
    assertEquals(call.name, "func")
    assertEquals(intrinsic.name, "threadIdx.x")
    assertEquals(load.index, ConstI32(0))
    assertEquals(SyncThreads, SyncThreads)
  }

  test("Stmt and Block structures") {
    val ptr = Param("ptr", Ty.Ptr(Ty.F32))
    val store = Store(ptr, ConstI32(0), ConstF32(1.0f))
    val ifStmt = IfThen(Lt(ConstI32(1), ConstI32(2)), Block(List(SyncThreadsStmt)))
    val block = Block(List(store, ifStmt))

    assertEquals(block.stmts.length, 2)
    assertEquals(ifStmt.thenBody.stmts.head, SyncThreadsStmt)
  }

  test("Kernel and Device Function definitions") {
    val p = Param("x", Ty.F32)
    val body = Block(Nil)
    val df = DeviceFuncDef("sq", List(p), body, Mul(p, p), Ty.F32)
    val k = KernelDef("k", List(p), body, sharedMems = List(SharedMem("s", Ty.F32, 128)), deviceFuncs = List(df))

    assertEquals(k.name, "k")
    assertEquals(k.sharedMems.head.name, "s")
    assertEquals(k.deviceFuncs.head.name, "sq")
  }

  test("Builder manual use") {
    val parent = new Builder()
    val child = new Builder(parent = Some(parent))
    
    val sm = SharedMem("s", Ty.I32, 10)
    child.addSharedMem(sm)
    assertEquals(parent.getSharedMems(), List(sm))
    assertEquals(child.getSharedMems(), Nil)
    
    val df = DeviceFuncDef("f", Nil, Block(Nil), ConstI32(0), Ty.I32)
    child.addDeviceFunc(df)
    assertEquals(parent.getDeviceFuncs(), List(df))
    assertEquals(child.getDeviceFuncs(), Nil)
    
    child.addStmt(SyncThreadsStmt)
    assertEquals(child.buildBlock().stmts, List(SyncThreadsStmt))
    assertEquals(parent.buildBlock().stmts, Nil)
  }
}
