package com.cuda4s.core.dsl

import scala.compiletime.testing.typeCheckErrors

import munit.FunSuite

class DslSafetySuite extends FunSuite:
  test("GPU conditions cannot be used as Scala Boolean values"):
    val errors = typeCheckErrors(
      """
        import com.cuda4s.core.dsl.CudaDsl.*
        val condition = threadIdx.x < literal(32)
        if condition then ()
      """
    )

    assert(errors.nonEmpty)

  test("read-only buffer elements cannot be assigned"):
    val errors = typeCheckErrors(
      """
        import com.cuda4s.core.dsl.CudaDsl.*
        val source = input[Float]("source")
        kernel("invalidWrite", source) {
          source(literal(0)) := literal(1.0f)
        }
      """
    )

    assert(errors.nonEmpty)

  test("GPU branches require GPU Boolean expressions"):
    val errors = typeCheckErrors(
      """
        import com.cuda4s.core.dsl.CudaDsl.*
        kernel("invalidCondition") {
          when(literal(1)) {}
        }
      """
    )

    assert(errors.nonEmpty)
