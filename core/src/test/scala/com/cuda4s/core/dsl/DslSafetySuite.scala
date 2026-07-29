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

  test("floating-point remainder is rejected"):
    val errors = typeCheckErrors(
      """
        import com.cuda4s.core.dsl.CudaDsl.*
        val invalid = literal(1.0f) % literal(0.5f)
      """
    )

    assert(errors.nonEmpty)

  test("F16 and BF16 support declared arithmetic capabilities"):
    val errors = typeCheckErrors(
      """
        import com.cuda4s.core.dsl.CudaDsl.*
        import com.cuda4s.core.types.*

        val f16a = literal(Float16.fromBits(0.toShort))
        val f16b = literal(Float16.fromBits(0.toShort))
        val bf16a = literal(BFloat16.fromBits(0.toShort))
        val bf16b = literal(BFloat16.fromBits(0.toShort))

        val f16Result = f16a + f16b
        val bf16Result = bf16a * bf16b
      """
    )

    assertEquals(errors, Nil)

  test("FP8 generic arithmetic is rejected"):
    val errors = typeCheckErrors(
      """
        import com.cuda4s.core.dsl.CudaDsl.*
        import com.cuda4s.core.types.*

        val left = literal(Float8E4M3.fromBits(0.toByte))
        val right = literal(Float8E4M3.fromBits(0.toByte))
        val invalid = left + right
      """
    )

    assert(errors.nonEmpty)

  test("FP8 conversion is explicit and type checks"):
    val errors = typeCheckErrors(
      """
        import com.cuda4s.core.dsl.CudaDsl.*

        val narrowed = convert.f32ToFP8E4M3(literal(1.0f))
        val restored = convert.fp8E4M3ToF32(narrowed)
      """
    )

    assertEquals(errors, Nil)

  test("low-precision values have declared accumulation types"):
    val errors = typeCheckErrors(
      """
        import com.cuda4s.core.dsl.CudaDsl.*
        import com.cuda4s.core.ir.Expr
        import com.cuda4s.core.types.*

        val f16: Expr[Float] =
          literal(Float16.fromBits(0.toShort)).toAccumulator
        val bf16: Expr[Float] =
          literal(BFloat16.fromBits(0.toShort)).toAccumulator
        val fp8: Expr[Float] =
          literal(Float8E5M2.fromBits(0.toByte)).toAccumulator
      """
    )

    assertEquals(errors, Nil)

  test("Boolean values cannot be accumulated"):
    val errors = typeCheckErrors(
      """
        import com.cuda4s.core.dsl.CudaDsl.*
        val invalid = literal(true).toAccumulator
      """
    )

    assert(errors.nonEmpty)

  test("reduction initial values must use the declared accumulator type"):
    val errors = typeCheckErrors(
      """
        import com.cuda4s.core.dsl.CudaDsl.*
        import com.cuda4s.core.ir.Expr
        import com.cuda4s.core.types.*

        val invalid: Expr[Float16] =
          reduceSum(
            "i",
            literal(0),
            literal(8),
            literal(Float16.fromBits(0.toShort))
          )(_ => literal(Float16.fromBits(0.toShort)))
      """
    )

    assert(errors.nonEmpty)
