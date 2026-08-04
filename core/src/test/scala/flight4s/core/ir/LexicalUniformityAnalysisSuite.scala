package flight4s.core.ir

import munit.FunSuite

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.types.*

class LexicalUniformityAnalysisSuite extends FunSuite:
  test("local declarations and accumulations update lexical uniformity"):
    val accumulator = LocalVariable("accumulator", F32)
    val source = BufferElement[Float, ReadOnly]("source", literal(0), F32)
    val declared = LocalDeclaration(accumulator, literal(0.0f))
    val accumulated =
      Accumulate(accumulator, Load(source), summon[AdditiveType[Float]])

    val afterDeclaration =
      UniformityAnalysis.scopeAfter(declared, UniformityScope.empty)
    val afterAccumulation =
      UniformityAnalysis.scopeAfter(accumulated, afterDeclaration)

    assertEquals(
      UniformityAnalysis.expression(Load(accumulator), afterDeclaration),
      Uniformity.GridUniform
    )
    assertEquals(
      UniformityAnalysis.expression(Load(accumulator), afterAccumulation),
      Uniformity.Varying
    )

  test("loop and reduction indexes inherit their bound uniformity"):
    val loop = ForLoop(
      LoopIndex("loopIndex"),
      literal(0),
      blockIdx.x,
      Block(Vector.empty)
    )
    val blockReduction: Expr[Int] =
      reduceSum("reductionIndex", literal(0), blockIdx.x, literal(0)) { index =>
        index
      }
    val varyingReduction: Expr[Int] =
      reduceSum("reductionIndex", literal(0), threadIdx.x, literal(0)) { index =>
        index
      }

    val loopScope = UniformityAnalysis.loopScope(loop, UniformityScope.empty)

    assertEquals(
      UniformityAnalysis.expression(loop.index, loopScope),
      Uniformity.BlockUniform
    )
    assertEquals(
      UniformityAnalysis.expression(blockReduction),
      Uniformity.BlockUniform
    )
    assertEquals(
      UniformityAnalysis.expression(varyingReduction),
      Uniformity.Varying
    )

  test("control-dependent local mutations remain conservative"):
    val branchDefinition = kernel("branchMutatesLocal", params()) { _ =>
      val accumulator = local("accumulator", literal(0.0f))
      when(threadIdx.x < literal(32)) {
        accumulate(accumulator, literal(1.0f))
      }
      when(accumulator.read === literal(0.0f)) {
        barrier()
      }
    }
    val loopDefinition = kernel("loopMutatesLocal", params()) { _ =>
      val accumulator = local("accumulator", literal(0.0f))
      gpuFor("index", literal(0), threadIdx.x) { _ =>
        accumulate(accumulator, literal(1.0f))
      }
      when(accumulator.read === literal(0.0f)) {
        barrier()
      }
    }

    assertEquals(
      KernelValidator.validate(branchDefinition).warnings.map(_.code),
      Vector(ValidationWarningCode.BarrierMayDiverge)
    )
    assertEquals(
      KernelValidator.validate(loopDefinition).warnings.map(_.code),
      Vector(ValidationWarningCode.BarrierMayDiverge)
    )
