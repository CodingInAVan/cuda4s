package com.cuda4s.jit.codegen

import com.cuda4s.jit.ir.*

import scala.collection.mutable

object CudaCodegen {
  def generate(kernel: KernelDef): String =
    val sb = new StringBuilder()
    val ctx = new GenerationContext(sb)

    // 1. Forward Declarations (Required for NVRTC if functions call each other or strictly ordered)
    kernel.deviceFuncs.foreach { df =>
      ctx.line(s"__device__ ${emitTy(df.returnTy)} ${df.name}(${emitParams(df.params)});")
    }
    if (kernel.deviceFuncs.nonEmpty) ctx.newline()

    // 2. Device Functions
    kernel.deviceFuncs.foreach { df =>
      ctx.line(s"__device__ ${emitTy(df.returnTy)} ${df.name}(${emitParams(df.params)}) {")
      ctx.indent()
      // Device functions have their own scope (params are declared)
      ctx.withScope(df.params.map(_.name)) {
        emitBlockBody(df.body, ctx)
        ctx.line(s"return ${emitExpr(df.returnExpr)};")
      }
      ctx.outdent()
      ctx.line("}")
      ctx.newline()
    }

    // 3. Kernel Definition
    ctx.line("extern \"C\" __global__")
    ctx.line(s"void ${kernel.name}(${emitParams(kernel.params, isKernel = true)}) {")
    ctx.indent()

    // Kernel Scope
    ctx.withScope(kernel.params.map(_.name)) {
      // Shared Memory
      kernel.sharedMems.foreach { sm =>
        ctx.line(s"__shared__ ${emitTy(sm.ty)} ${sm.name}[${sm.size}];")
      }

      // Body
      emitBlockBody(kernel.body, ctx)
    }

    ctx.outdent()
    ctx.line("}")

    sb.toString()


  private def emitBlockBody(block: Block, ctx: GenerationContext): Unit =
    block.stmts.foreach(stmt => emitStmt(stmt, ctx))

  private def emitStmt(stmt: Stmt, ctx: GenerationContext): Unit = stmt match
    case Store(ptr, index, value) =>
      ctx.line(s"${emitExpr(ptr)}[${emitExpr(index)}] = ${emitExpr(value)};")

    case Assign(lhs, rhs) =>
      val name = emitExpr(lhs)
      if (ctx.isDeclared(name)) then
        // Variable already exists in this scope (or parent), just assign
        ctx.line(s"$name = ${emitExpr(rhs)};")
      else
        // New variable definition
        // We rely on the IR Param node to carry the type info for the definition
        val typeStr = lhs match
          case p: Param => emitTy(p.ty)
          case _        => "auto" // Fallback, though IR should essentially always use Param for LHS

        ctx.line(s"$typeStr $name = ${emitExpr(rhs)};")
        ctx.declare(name)

    case IfThen(cond, thenBody) =>
      ctx.line(s"if (${emitExpr(cond)}) {")
      ctx.indent()
      ctx.withScope() { emitBlockBody(thenBody, ctx) }
      ctx.outdent()
      ctx.line("}")

    case IfThenElse(cond, thenBody, elseBody) =>
      ctx.line(s"if (${emitExpr(cond)}) {")
      ctx.indent()
      ctx.withScope() { emitBlockBody(thenBody, ctx) }
      ctx.outdent()
      ctx.line("} else {")
      ctx.indent()
      ctx.withScope() { emitBlockBody(elseBody, ctx) }
      ctx.outdent()
      ctx.line("}")

    case While(cond, body) =>
      ctx.line(s"while (${emitExpr(cond)}) {")
      ctx.indent()
      ctx.withScope() { emitBlockBody(body, ctx) }
      ctx.outdent()
      ctx.line("}")

    case For(init, cond, step, body) =>
      ctx.withScope() {
        val initStr = init match
          case Assign(lhs, rhs) =>
            val name = emitExpr(lhs)
            // Always prefer declaring loop vars inline: for(int i=0;...)
            val ty = lhs match { case p: Param => emitTy(p.ty); case _ => "auto" }
            ctx.declare(name) // It is now declared in this inner scope
            s"$ty $name = ${emitExpr(rhs)}"
          case other =>
            // Fallback for complex init, though uncommon in this IR
            "/* complex init */"

        val stepStr = step match
          case Assign(lhs, rhs) => s"${emitExpr(lhs)} = ${emitExpr(rhs)}"
          // Handle implicit expression-statements (like i++ represented as i = i + 1)
          case _ => ""

        ctx.line(s"for ($initStr; ${emitExpr(cond)}; $stepStr) {")
        ctx.indent()
        emitBlockBody(body, ctx)
        ctx.outdent()
        ctx.line("}")
      }

    case BlockStmt(block) =>
      ctx.line("{")
      ctx.indent()
      ctx.withScope() { emitBlockBody(block, ctx) }
      ctx.outdent()
      ctx.line("}")

    case SyncThreadsStmt =>
      ctx.line("__syncthreads();")

  private def emitExpr(expr: Expr): String = expr match {
    case Param(name, _, _) => name
    case ConstI32(v)    => v.toString
    case ConstU32(v)    => s"${v}u"
    case ConstF32(v)    => s"${v}f"
    case ConstF64(v)    => v.toString

    case Add(l, r)      => s"(${emitExpr(l)} + ${emitExpr(r)})"
    case Sub(l, r)      => s"(${emitExpr(l)} - ${emitExpr(r)})"
    case Mul(l, r)      => s"(${emitExpr(l)} * ${emitExpr(r)})"
    case Div(l, r)      => s"(${emitExpr(l)} / ${emitExpr(r)})"
    case Mod(l, r)      => s"(${emitExpr(l)} % ${emitExpr(r)})"

    case Lt(l, r)       => s"(${emitExpr(l)} < ${emitExpr(r)})"
    case Gt(l, r)       => s"(${emitExpr(l)} > ${emitExpr(r)})"
    case Le(l, r)       => s"(${emitExpr(l)} <= ${emitExpr(r)})"
    case Ge(l, r)       => s"(${emitExpr(l)} >= ${emitExpr(r)})"
    case Eq(l, r)       => s"(${emitExpr(l)} == ${emitExpr(r)})"
    case Ne(l, r)       => s"(${emitExpr(l)} != ${emitExpr(r)})"
    case And(l, r)      => s"(${emitExpr(l)} && ${emitExpr(r)})"
    case Or(l, r)       => s"(${emitExpr(l)} || ${emitExpr(r)})"

    case Call(name, args) =>
      s"$name(${args.map(emitExpr).mkString(", ")})"

    case Intrinsic(name, _) => name

    case Load(ptr, idx) => s"${emitExpr(ptr)}[${emitExpr(idx)}]"

    case SyncThreads    => "__syncthreads()"
  }

  private def emitParams(params: List[Param], isKernel: Boolean = false): String =
    params.map { p =>
      val qual = if (isKernel && p.isConstant) "__constant__ " else ""
      s"$qual${emitTy(p.ty)} ${p.name}"
    }.mkString(", ")

  private def emitTy(ty: Ty): String = ty match
    case Ty.I32 => "int"
    case Ty.U32 => "unsigned int"
    case Ty.F32 => "float"
    case Ty.F64 => "double"
    case Ty.Ptr(inner) => s"${emitTy(inner)}*"
}
