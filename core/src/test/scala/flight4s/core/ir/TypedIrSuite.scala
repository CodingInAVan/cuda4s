package flight4s.core.ir

import munit.FunSuite

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.types.*

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
    val signature = params(result)

    val definition = kernel("writeResult", signature) { bindings =>
      val boundResult = bindings.head
      when(index < literal(32)) {
        boundResult(index) := literal(1.0f)
        barrier()
      }
    }

    assertEquals(definition.params, Vector(result))
    assertEquals(definition.body.statements.size, 1)

    val branch = definition.body.statements.head.asInstanceOf[IfThen]
    assertEquals(branch.thenBlock.statements.size, 2)
    assert(branch.thenBlock.statements.head.isInstanceOf[Store[?, ?]])
    val barrierStatement =
      branch.thenBlock.statements(1).asInstanceOf[Barrier]
    assertNotEquals(barrierStatement.span, SourceSpan.Unknown)

  test("read-only buffer elements can be loaded"):
    val source = input[Float]("source")
    val loaded = source(literal(0)).read

    assertEquals(loaded.valueType, F32)
    assert(loaded.isInstanceOf[Load[?, ?, ?]])

  test("F32 to F16 conversion retains rounding intent"):
    val converted = convert
      .f32ToF16(literal(1.5f), RoundingMode.TowardZero)
      .asInstanceOf[Convert[Float, Float16]]

    assertEquals(converted.valueType, F16)
    assertEquals(converted.rounding, RoundingMode.TowardZero)
    assertEquals(converted.saturation, SaturationMode.NoSaturation)

  test("FP8 conversion retains format and saturation intent"):
    val e4m3 = convert
      .f32ToFP8E4M3(literal(1.0f))
      .asInstanceOf[Convert[Float, Float8E4M3]]
    val e5m2 = convert
      .f32ToFP8E5M2(literal(1.0f), SaturationMode.NoSaturation)
      .asInstanceOf[Convert[Float, Float8E5M2]]

    assertEquals(e4m3.valueType, FP8E4M3)
    assertEquals(e4m3.saturation, SaturationMode.SaturateFinite)
    assertEquals(e5m2.valueType, FP8E5M2)
    assertEquals(e5m2.saturation, SaturationMode.NoSaturation)

  test("low-precision values retain explicit accumulation intent"):
    val f16Accumulation: Expr[Float] =
      literal(Float16.fromBits(0.toShort)).toAccumulator
    val fp8Accumulation: Expr[Float] =
      literal(Float8E4M3.fromBits(0.toByte)).toAccumulator

    val f16Node = f16Accumulation.asInstanceOf[ToAccumulator[Float16, Float]]
    val fp8Node = fp8Accumulation.asInstanceOf[ToAccumulator[Float8E4M3, Float]]

    assertEquals(f16Node.rule.inputType, F16)
    assertEquals(f16Node.valueType, F32)
    assertEquals(fp8Node.rule.inputType, FP8E4M3)
    assertEquals(fp8Node.valueType, F32)
