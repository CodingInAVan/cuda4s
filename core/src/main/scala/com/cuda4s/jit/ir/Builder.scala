package com.cuda4s.jit.ir

import com.cuda4s.jit.ir.DSL.SVal

import scala.collection.mutable.ListBuffer

class Builder(
    private val parent: Option[Builder] = None,
    private val stmts: ListBuffer[Stmt] = ListBuffer[Stmt](),
    private val sharedMems: ListBuffer[SharedMem] = ListBuffer[SharedMem](),
    private val deviceFuncs: ListBuffer[DeviceFuncDef] = ListBuffer[DeviceFuncDef]()
):
  def addStmt(stmt: Stmt): Unit =
    stmts += stmt

  def addSharedMem(sm: SharedMem): Unit =
    if (parent.isDefined) parent.get.addSharedMem(sm)
    else sharedMems += sm

  def addDeviceFunc(df: DeviceFuncDef): Unit =
    if (parent.isDefined) parent.get.addDeviceFunc(df)
    else deviceFuncs += df

  def buildBlock(): Block =
    Block(stmts.toList)
    
  def removeLastStmt(): Stmt =
    stmts.remove(stmts.size - 1)

  def getSharedMems(): List[SharedMem] = sharedMems.toList
  def getDeviceFuncs(): List[DeviceFuncDef] = deviceFuncs.toList

  inline def forLoop(inline init: Stmt, inline cond: SVal, inline step: Stmt)(inline body: Builder ?=> Unit): Unit =
    val nestedBuilder = new Builder(parent = Some(this))
    body(using nestedBuilder)
    addStmt(For(init, cond.expr, step, nestedBuilder.buildBlock()))

  inline def ifThen(inline cond: Expr)(inline body: Builder ?=> Unit): Unit =
    val nestedBuilder = new Builder(parent = Some(this))
    body(using nestedBuilder)
    addStmt(IfThen(cond, nestedBuilder.buildBlock()))

  inline def ifThenElse(inline cond: Expr)(inline thenBody: Builder ?=> Unit, inline elseBody: Builder ?=> Unit): Unit =
    val thenBuilder = new Builder(parent = Some(this))
    thenBody(using thenBuilder)
    val elseBuilder = new Builder(parent = Some(this))
    elseBody(using elseBuilder)
    addStmt(IfThenElse(cond, thenBuilder.buildBlock(), elseBuilder.buildBlock()))
