package com.cuda4s.core.ir

import munit.FunSuite

import com.cuda4s.core.dsl.CudaDsl.*
import com.cuda4s.core.types.*

class ReductionSuite extends FunSuite:
  test("reduceSum preserves one symbolic body instead of unrolling expressions"):
    val source = input[Float16]("source")
    val reduction: Expr[Float] =
      reduceSum(
        indexName = "i",
        from = literal(0),
        until = literal(1024),
        initial = literal(0.0f),
        policy = ReductionPolicy.Fast
      ) { index =>
        source(index).read
      }

    val node = reduction.asInstanceOf[ReduceSum[Float16, Float]]

    assertEquals(node.index.name, "i")
    assertEquals(node.from, literal(0))
    assertEquals(node.until, literal(1024))
    assertEquals(node.rule.inputType, F16)
    assertEquals(node.valueType, F32)
    assertEquals(node.policy, ReductionPolicy.Fast)
    assert(node.value.isInstanceOf[Load[?, ?, ?]])

  test("a reduction index is scoped to its symbolic body"):
    val source = input[Float16]("source")
    val result = output[Float]("result")
    val sum: Expr[Float] =
      reduceSum("i", literal(0), literal(64), literal(0.0f)) { index =>
        source(index).read
      }
    val definition = kernel("sumValues", source, result) {
      result(literal(0)) := sum
    }

    assert(KernelValidator.validate(definition).isValid)

  test("an unbound reduction index is rejected"):
    val result = output[Float]("result")
    val index = ReductionIndex("i")
    val definition = kernel("unboundIndex", result) {
      result(index) := literal(1.0f)
    }

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.UnboundReductionIndex))

  test("nested reductions cannot shadow an active reduction index"):
    val result = output[Float]("result")
    val nested: Expr[Float] =
      reduceSum("i", literal(0), literal(4), literal(0.0f)) { _ =>
        reduceSum("i", literal(0), literal(4), literal(0.0f)) { _ =>
          literal(1.0f)
        }
      }
    val definition = kernel("shadowedIndex", result) {
      result(literal(0)) := nested
    }

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.DuplicateReductionIndex))

  test("reduction indices cannot conflict with kernel parameters"):
    val source = input[Float]("i")
    val result = output[Float]("result")
    val sum: Expr[Float] =
      reduceSum("i", literal(0), literal(4), literal(0.0f)) { index =>
        source(index).read
      }
    val definition = kernel("conflictingIndex", source, result) {
      result(literal(0)) := sum
    }

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.ReductionIndexConflictsWithParameter))

  test("reduction policy remains explicit in the IR"):
    def reduction(policy: ReductionPolicy): ReduceSum[Float, Float] =
      reduceSum(
        "i",
        literal(0),
        literal(8),
        literal(0.0f),
        policy
      )(_ => literal(1.0f)).asInstanceOf[ReduceSum[Float, Float]]

    assertEquals(reduction(ReductionPolicy.Strict).policy, ReductionPolicy.Strict)
    assertEquals(
      reduction(ReductionPolicy.Deterministic).policy,
      ReductionPolicy.Deterministic
    )
    assertEquals(reduction(ReductionPolicy.Fast).policy, ReductionPolicy.Fast)
