package com.cuda4s.core.ir

import munit.FunSuite

import com.cuda4s.core.dsl.CudaDsl.*
import com.cuda4s.core.types.*

class TypedIrSuite extends FunSuite:
  test("numeric expressions retain their CUDA type"):
    val expression = literal(2.0f) * literal(4.0f) + literal(1.0f)

    assertEquals(expression.valueType, F32)
    assert(expression.isInstanceOf[Binary[Float]])

  test("comparison expressions produce a GPU boolean expression"):
    val condition = threadIdx.x < literal(32)

    assertEquals(condition.valueType, Bool)
    assert(condition.isInstanceOf[Compare[?]])

  test("explicit GPU control flow builds statements"):
    val result = output[Float]("result")
    val index = threadIdx.x

    val definition = kernel("writeResult", result) {
      when(index < literal(32)) {
        result(index) := literal(1.0f)
        barrier()
      }
    }

    assertEquals(definition.params, Vector(result))
    assertEquals(definition.body.statements.size, 1)

    val branch = definition.body.statements.head.asInstanceOf[IfThen]
    assertEquals(branch.thenBlock.statements.size, 2)
    assert(branch.thenBlock.statements.head.isInstanceOf[Store[?, ?]])
    assertEquals(branch.thenBlock.statements(1), Barrier())

  test("read-only buffer elements can be loaded"):
    val source = input[Float]("source")
    val loaded = source(literal(0)).read

    assertEquals(loaded.valueType, F32)
    assert(loaded.isInstanceOf[Load[?, ?, ?]])
