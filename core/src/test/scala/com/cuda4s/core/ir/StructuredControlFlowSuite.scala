package com.cuda4s.core.ir

import munit.FunSuite

import com.cuda4s.core.dsl.CudaDsl.*
import com.cuda4s.core.types.*

class StructuredControlFlowSuite extends FunSuite:
  test("local accumulation inside a unit-stride GPU loop builds structured IR"):
    val source = input[Float16]("source")
    val result = output[Float]("result")
    val definition = kernel("sumLoop", source, result) {
      val accumulator = local("accumulator", literal(0.0f))

      gpuFor("i", literal(0), literal(32)) { index =>
        accumulate(accumulator, source(index).read.toAccumulator)
      }

      result(literal(0)) := accumulator.read
    }

    assert(KernelValidator.validate(definition).isValid)
    assertEquals(definition.body.statements.size, 3)
    assert(definition.body.statements.head.isInstanceOf[LocalDeclaration[?]])

    val loop = definition.body.statements(1).asInstanceOf[ForLoop]
    assertEquals(loop.index.name, "i")
    assertEquals(loop.body.statements.size, 1)
    assert(loop.body.statements.head.isInstanceOf[Accumulate[?]])

  test("locals cannot be used before their declaration"):
    val accumulator = LocalVariable("accumulator", F32)
    val definition = KernelIR(
      "useBeforeDeclaration",
      Vector.empty,
      Block(
        Vector(
          Store(accumulator, literal(1.0f)),
          LocalDeclaration(accumulator, literal(0.0f))
        )
      )
    )

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.UnboundLocal))

  test("locals declared in a branch do not escape the branch"):
    val temporary = LocalVariable("temporary", F32)
    val definition = KernelIR(
      "branchScope",
      Vector.empty,
      Block(
        Vector(
          IfThen(
            literal(true),
            Block(Vector(LocalDeclaration(temporary, literal(0.0f))))
          ),
          Store(temporary, literal(1.0f))
        )
      )
    )

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.UnboundLocal))

  test("loop indices do not escape their loop body"):
    val result = output[Float]("result")
    val index = LoopIndex("i")
    val definition = KernelIR(
      "loopScope",
      Vector(result),
      Block(
        Vector(
          ForLoop(index, literal(0), literal(4), Block(Vector.empty)),
          Store(
            BufferElement[Float, ReadWrite]("result", index, F32),
            literal(1.0f)
          )
        )
      )
    )

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.UnboundLoopIndex))

  test("local and loop names cannot conflict with active bindings"):
    val parameter = value[Float]("accumulator")
    val localValue = LocalVariable("accumulator", F32)
    val loopIndex = LoopIndex("accumulator")
    val definition = KernelIR(
      "bindingConflicts",
      Vector(parameter),
      Block(
        Vector(
          LocalDeclaration(localValue, literal(0.0f)),
          ForLoop(loopIndex, literal(0), literal(4), Block(Vector.empty))
        )
      )
    )

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.LocalNameConflictsWithBinding))
    assert(codes.contains(ValidationCode.LoopIndexConflictsWithBinding))
