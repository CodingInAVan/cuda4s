package flight4s.core.ir

import munit.FunSuite

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.types.*

class EffectAnalysisSuite extends FunSuite:
  test("pure scalar expressions have no effects"):
    val effect = EffectAnalysis.expression(literal(2.0f) + literal(4.0f))

    assertEquals(effect, EffectSummary.empty)
    assert(effect.isPure)

  test("loads record their memory space and effects needed to address them"):
    val offset = Load(ConstantElement("offsets", literal(0), I32))
    val inputElement = BufferElement[Float, ReadOnly]("input", offset, F32)

    val effect = EffectAnalysis.expression(Load(inputElement))

    assertEquals(
      effect.readSpaces,
      Set(EffectMemorySpace.Constant, EffectMemorySpace.Global)
    )
    assertEquals(effect.writtenSpaces, Set.empty)
    assert(!effect.hasBarrier)
    assert(!effect.isPure)

  test("blocks aggregate memory effects and barriers through structured control flow"):
    val accumulator = LocalVariable("accumulator", F32)
    val coefficient = ConstantElement("coefficients", literal(0), F32)
    val tile = SharedElement("tile", Vector(literal(0), literal(0)), F32)
    val outputElement = BufferElement[Float, ReadWrite]("output", literal(0), F32)
    val index = LoopIndex("i")

    val block = Block(
      Vector(
        LocalDeclaration(accumulator, literal(0.0f)),
        IfThen(
          literal(true),
          Block(
            Vector(
              Accumulate(
                accumulator,
                Load(coefficient),
                summon[AdditiveType[Float]]
              ),
              Store(tile, Load(accumulator)),
              Barrier()
            )
          )
        ),
        ForLoop(
          index,
          literal(0),
          literal(8),
          Block(Vector(Store(outputElement, Load(accumulator))))
        )
      )
    )

    val effect = EffectAnalysis.block(block)

    assertEquals(
      effect.readSpaces,
      Set(EffectMemorySpace.Constant, EffectMemorySpace.Local)
    )
    assertEquals(
      effect.writtenSpaces,
      Set(
        EffectMemorySpace.Global,
        EffectMemorySpace.Shared,
        EffectMemorySpace.Local
      )
    )
    assert(effect.hasBarrier)

  test("scoped blocks expose their nested effects"):
    val outputElement = BufferElement[Float, ReadWrite]("output", literal(0), F32)
    val scoped = ScopedBlock(
      Block(Vector(Store(outputElement, literal(1.0f)), Barrier()))
    )

    val effect = EffectAnalysis.statement(scoped)

    assertEquals(effect.readSpaces, Set.empty)
    assertEquals(effect.writtenSpaces, Set(EffectMemorySpace.Global))
    assert(effect.hasBarrier)
