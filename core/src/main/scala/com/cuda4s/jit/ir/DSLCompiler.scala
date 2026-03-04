package com.cuda4s.jit.ir

object DSLCompiler:
  def compileDSL(name: String, params: List[Param])(body: Builder ?=> Unit): KernelDef =
    val builder = new Builder()
    body(using builder)
    KernelDef(name, params, builder.buildBlock(), builder.getSharedMems(), builder.getDeviceFuncs())

  def deviceFunc(name: String, params: List[Param], returnTy: Ty)(body: Builder ?=> Expr)(using b: Builder): String =
    val dfBuilder = new Builder(parent = Some(b))
    val resExpr = body(using dfBuilder)
    val df = DeviceFuncDef(name, params, dfBuilder.buildBlock(), resExpr, returnTy)
    b.addDeviceFunc(df)
    name
