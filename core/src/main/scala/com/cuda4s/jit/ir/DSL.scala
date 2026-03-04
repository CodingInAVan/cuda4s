package com.cuda4s.jit.ir

import scala.language.implicitConversions

object DSL:
  case class SVal(expr: Expr):
    inline def toBoolean: Boolean = true // Dummy for if() support in macro
    inline def +(inline other: SVal): SVal = SVal(Add(this.expr, other.expr))
    inline def -(inline other: SVal): SVal = SVal(Sub(this.expr, other.expr))
    inline def *(inline other: SVal): SVal = SVal(Mul(this.expr, other.expr))
    inline def /(inline other: SVal): SVal = SVal(Div(this.expr, other.expr))
    inline def %(inline other: SVal): SVal = SVal(Mod(this.expr, other.expr))
    inline def :=(inline value: SVal)(using b: Builder): Unit =
      b.addStmt(Assign(this.expr, value.expr))

    inline def <(inline other: SVal): SVal = SVal(Lt(this.expr, other.expr))
    inline def >(inline other: SVal): SVal = SVal(Gt(this.expr, other.expr))
    inline def <=(inline other: SVal): SVal = SVal(Le(this.expr, other.expr))
    inline def >=(inline other: SVal): SVal = SVal(Ge(this.expr, other.expr))
    inline def ===(inline other: SVal): SVal = SVal(Eq(this.expr, other.expr))
    inline def !==(inline other: SVal): SVal = SVal(Ne(this.expr, other.expr))
    inline def &&(inline other: SVal): SVal = SVal(And(this.expr, other.expr))
    inline def ||(inline other: SVal): SVal = SVal(Or(this.expr, other.expr))

  case class SPtr(expr: Expr):
    inline def apply(inline index: SVal): SVal = SVal(Load(expr, index.expr))
    inline def update(inline index: SVal, inline value: SVal)(using b: Builder): Unit =
      b.addStmt(Store(expr, index.expr, value.expr))

    inline def :=(inline value: SVal)(using b: Builder): Unit =
      b.addStmt(Assign(expr, value.expr))

    inline def unary_* : SVal = apply(0)

  extension (p: SPtr)
    inline def +(inline offset: SVal): SPtr = SPtr(Add(p.expr, offset.expr))
    inline def -(inline offset: SVal): SPtr = SPtr(Sub(p.expr, offset.expr))
    inline def at(inline index: SVal): SVal = p.apply(index)

  type *[T] = SPtr

  given Conversion[Int, SVal] = i => SVal(ConstI32(i))
  given Conversion[Float, SVal] = f => SVal(ConstF32(f))
  given Conversion[Param, SVal] = p => SVal(p)
  given Conversion[Param, SPtr] = p => SPtr(p)
  given Conversion[SPtr, SVal] = p => p.unary_*

  extension (inline p: Param)
    inline def <(inline other: Param): Boolean = true
    inline def >(inline other: Param): Boolean = true
    inline def <=(inline other: Param): Boolean = true
    inline def >=(inline other: Param): Boolean = true
    inline def ===(inline other: Param): Boolean = true
    inline def !==(inline other: Param): Boolean = true

  extension (inline i: Int)
    inline def <(inline other: Param): Boolean = true
    inline def >(inline other: Param): Boolean = true
    inline def <=(inline other: Param): Boolean = true
    inline def >=(inline other: Param): Boolean = true
    inline def ===(inline other: Param): Boolean = true
    inline def !==(inline other: Param): Boolean = true
    inline def s: SVal = SVal(ConstI32(i))
    inline def u: SVal = SVal(ConstU32(i))
  extension (inline f: Float) inline def s: SVal = SVal(ConstF32(f))
  extension (inline d: Double) inline def s: SVal = SVal(ConstF64(d))

  object threadIdx:
    inline def x: SVal = SVal(Intrinsic("threadIdx.x", Ty.I32))
    inline def y: SVal = SVal(Intrinsic("threadIdx.y", Ty.I32))
    inline def z: SVal = SVal(Intrinsic("threadIdx.z", Ty.I32))

  object blockIdx:
    inline def x: SVal = SVal(Intrinsic("blockIdx.x", Ty.I32))
    inline def y: SVal = SVal(Intrinsic("blockIdx.y", Ty.I32))
    inline def z: SVal = SVal(Intrinsic("blockIdx.z", Ty.I32))

  object blockDim:
    inline def x: SVal = SVal(Intrinsic("blockDim.x", Ty.I32))
    inline def y: SVal = SVal(Intrinsic("blockDim.y", Ty.I32))
    inline def z: SVal = SVal(Intrinsic("blockDim.z", Ty.I32))

  inline def syncThreads()(using b: Builder): Unit = b.addStmt(SyncThreadsStmt)

  inline def call(name: String, args: SVal*)(using b: Builder): SVal =
    SVal(Call(name, args.map(_.expr).toList))

  inline def sharedMem(name: String, ty: Ty, size: Int)(using b: Builder): SPtr =
    val sm = SharedMem(name, ty, size)
    b.addSharedMem(sm)
    SPtr(Param(name, Ty.Ptr(ty)))

  inline def ifThen(inline cond: SVal)(inline body: Builder ?=> Unit)(using b: Builder): Unit =
    b.ifThen(cond.expr)(body)

  inline def ifThenElse(inline cond: SVal)(inline thenBody: Builder ?=> Unit)(inline elseBody: Builder ?=> Unit)(using b: Builder): Unit =
    b.ifThenElse(cond.expr)(thenBody, elseBody)

  inline def forLoopStmt(inline init: Stmt, inline cond: SVal, inline step: Stmt)(inline body: Builder ?=> Unit)(using b: Builder): Unit =
    b.forLoop(init, cond, step)(body)

  inline def forRange(inline start: SVal, inline end: SVal)(inline body: SVal => Builder ?=> Unit)(using b: Builder): Unit =
    val name = s"_j${b.buildBlock().stmts.size}"
    val j = Param(name, Ty.I32)
    val init = Assign(j, start.expr)
    val cond = SVal(Lt(j, end.expr))
    val step = Assign(j, Add(j, ConstI32(1)))
    val nestedBuilder = new Builder(parent = Some(b))
    body(SVal(j))(using nestedBuilder)
    b.addStmt(For(init, cond.expr, step, nestedBuilder.buildBlock()))

  inline def varDef(name: String, ty: Ty, init: SVal)(using b: Builder): SVal =
    val p = Param(name, ty)
    b.addStmt(Assign(p, init.expr))
    SVal(p)

  inline def assign(lhs: SVal, rhs: SVal)(using b: Builder): Unit =
    b.addStmt(Assign(lhs.expr, rhs.expr))
