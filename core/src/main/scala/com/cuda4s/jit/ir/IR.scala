package com.cuda4s.jit.ir

sealed trait Expr
case class Param(name: String, ty: Ty, isConstant: Boolean = false) extends Expr
case class ConstI32(value: Int) extends Expr
case class ConstU32(value: Int) extends Expr
case class ConstF32(value: Float) extends Expr
case class ConstF64(value: Double) extends Expr
case class Add(lhs: Expr, rhs: Expr) extends Expr
case class Sub(lhs: Expr, rhs: Expr) extends Expr
case class Mul(lhs: Expr, rhs: Expr) extends Expr
case class Div(lhs: Expr, rhs: Expr) extends Expr
case class Mod(lhs: Expr, rhs: Expr) extends Expr
case class Lt(lhs: Expr, rhs: Expr) extends Expr
case class Gt(lhs: Expr, rhs: Expr) extends Expr
case class Le(lhs: Expr, rhs: Expr) extends Expr
case class Ge(lhs: Expr, rhs: Expr) extends Expr
case class Eq(lhs: Expr, rhs: Expr) extends Expr
case class Ne(lhs: Expr, rhs: Expr) extends Expr
case class And(lhs: Expr, rhs: Expr) extends Expr
case class Or(lhs: Expr, rhs: Expr) extends Expr
case class Call(name: String, args: List[Expr]) extends Expr
case class Intrinsic(name: String, ty: Ty) extends Expr
case object SyncThreads extends Expr
case class Load(ptr: Expr, index: Expr) extends Expr
case class For(init: Stmt, cond: Expr, step: Stmt, body: Block) extends Stmt

sealed trait Stmt
case class Store(ptr: Expr, index: Expr, value: Expr) extends Stmt
case class IfThen(cond: Expr, thenBody: Block) extends Stmt
case class IfThenElse(cond: Expr, thenBody: Block, elseBody: Block) extends Stmt
case class While(cond: Expr, body: Block) extends Stmt
case class BlockStmt(block: Block) extends Stmt
case class Assign(lhs: Expr, rhs: Expr) extends Stmt
case object SyncThreadsStmt extends Stmt

case class Block(stmts: List[Stmt])

case class SharedMem(name: String, ty: Ty, size: Int)

case class DeviceFuncDef(name: String, params: List[Param], body: Block, returnExpr: Expr, returnTy: Ty)

case class KernelDef(name: String, params: List[Param], body: Block, sharedMems: List[SharedMem] = Nil, deviceFuncs: List[DeviceFuncDef] = Nil)
