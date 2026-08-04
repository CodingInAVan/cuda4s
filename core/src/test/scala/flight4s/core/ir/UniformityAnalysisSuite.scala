package flight4s.core.ir

import munit.FunSuite

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.types.*

class UniformityAnalysisSuite extends FunSuite:
  test("literals, scalar parameters, and block dimensions are grid-uniform"):
    val scale = value[Float]("scale")

    assertEquals(UniformityAnalysis.expression(literal(1.0f)), Uniformity.GridUniform)
    assertEquals(UniformityAnalysis.expression(scale), Uniformity.GridUniform)
    assertEquals(UniformityAnalysis.expression(blockDim.x), Uniformity.GridUniform)

  test("intrinsics and composed expressions retain their least-uniform scope"):
    val withinBlock = blockIdx.x + literal(1)
    val withinThread = blockIdx.x + threadIdx.x

    assertEquals(UniformityAnalysis.expression(blockIdx.x), Uniformity.BlockUniform)
    assertEquals(UniformityAnalysis.expression(threadIdx.x), Uniformity.Varying)
    assertEquals(UniformityAnalysis.expression(withinBlock), Uniformity.BlockUniform)
    assertEquals(UniformityAnalysis.expression(withinThread), Uniformity.Varying)

  test("constant loads follow their index while mutable and private loads are conservative"):
    val coefficients = ConstantElement("coefficients", blockIdx.x, F32)
    val threadCoefficients = ConstantElement("coefficients", threadIdx.x, F32)
    val source = input[Float]("source")
    val local = LocalVariable("temporary", F32)

    assertEquals(
      UniformityAnalysis.expression(Load(coefficients)),
      Uniformity.BlockUniform
    )
    assertEquals(
      UniformityAnalysis.expression(Load(threadCoefficients)),
      Uniformity.Varying
    )
    assertEquals(
      UniformityAnalysis.expression(source(literal(0)).read),
      Uniformity.Varying
    )
    assertEquals(
      UniformityAnalysis.expression(Load(local)),
      Uniformity.Unknown
    )

  test("unrecognized intrinsics and lexically scoped indexes remain unknown"):
    val unsupportedIntrinsic = Intrinsic("warpIdx.x", I32)
    val unknownComposite = unsupportedIntrinsic + blockIdx.x

    assertEquals(
      UniformityAnalysis.expression(unsupportedIntrinsic),
      Uniformity.Unknown
    )
    assertEquals(
      UniformityAnalysis.expression(unknownComposite),
      Uniformity.Unknown
    )
    assertEquals(
      UniformityAnalysis.expression(LoopIndex("index")),
      Uniformity.Unknown
    )
    assertEquals(
      UniformityAnalysis.expression(ReductionIndex("reductionIndex")),
      Uniformity.Unknown
    )
