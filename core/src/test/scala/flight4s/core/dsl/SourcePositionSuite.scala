package flight4s.core.dsl

import munit.FunSuite

import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*

class SourcePositionSuite extends FunSuite:
  private def assertCaptured(spans: Iterable[SourceSpan]): Unit =
    spans.foreach { span =>
      assertNotEquals(span, SourceSpan.Unknown)
      assert(span.file.replace('\\', '/').endsWith("SourcePositionSuite.scala"))
      assert(span.startLine > 0)
      assert(span.startColumn > 0)
      assert(span.endLine >= span.startLine)
      assert(span.endColumn > 0)
    }

  test("DSL statement boundaries capture Scala call-site positions"):
    val outputParam = output[Float]("output")
    val definition = kernel("capturedPositions", params(outputParam)) {
      bindings =>
        val output = bindings.head
        val scratch = dynamicSharedArray[Float]("scratch")
        val index = local("index", threadIdx.x)
        when(index.read < blockDim.x) {
          output(index.read) := scratch(index.read).read
        }
        barrier()
    }

    val sharedSpan = definition.ir.sharedMemory.head.span
    val localDeclaration =
      definition.ir.body.statements(0).asInstanceOf[LocalDeclaration[Int]]
    val branch = definition.ir.body.statements(1).asInstanceOf[IfThen]
    val store = branch.thenBlock.statements.head.asInstanceOf[Store[?, ?]]
    val barrierStatement =
      definition.ir.body.statements(2).asInstanceOf[Barrier]
    val spans = Vector(
      sharedSpan,
      localDeclaration.span,
      branch.span,
      store.span,
      barrierStatement.span
    )

    assertCaptured(spans)
    assert(sharedSpan.startLine < localDeclaration.span.startLine)
    assert(localDeclaration.span.startLine < store.span.startLine)
    assert(store.span.startLine <= branch.span.startLine)
    assert(branch.span.startLine < barrierStatement.span.startLine)

    val generated = CudaCodegen.generate(definition).toOption.get
    assertEquals(
      generated.sourceMap.entries.map(_.sourceSpan),
      spans
    )

  test("all capturable declaration and execution boundaries retain positions"):
    val coefficients = constantArray[Float]("coefficients", 4)
    val outputParam = output[Float]("output")
    val definition = kernel("capturedBoundaries", params(outputParam)) {
      bindings =>
        val output = bindings.head
        sharedArray[Float]("line", 4)
        sharedArray2D[Float]("tile", 2, 2)
        sharedArray3D[Float]("cube", 2, 2, 2)
        val scratch = localArray[Float]("scratch", 2)
        val accumulator = local("accumulator", literal(0.0f))
        scratch(literal(0)) := literal(1.0f)
        gpuFor("i", literal(0), literal(2)) { index =>
          accumulate(accumulator, scratch(index).read)
        }
        gpuIf(threadIdx.x < literal(1)) {
          scratch(literal(0)) := literal(2.0f)
        } {
          scratch(literal(0)) := literal(3.0f)
        }
        val sum = reduceSum(
          "j",
          literal(0),
          literal(4),
          literal(0.0f)
        ) { index =>
          coefficients(index).read
        }
        output(literal(0)) := sum + accumulator.read
    }

    val statements = definition.ir.body.statements
    val localArrayDeclaration =
      statements(0).asInstanceOf[LocalArrayDeclaration[Float]]
    val localDeclaration =
      statements(1).asInstanceOf[LocalDeclaration[Float]]
    val firstStore = statements(2).asInstanceOf[Store[?, ?]]
    val loop = statements(3).asInstanceOf[ForLoop]
    val accumulation =
      loop.body.statements.head.asInstanceOf[Accumulate[Float]]
    val branch = statements(4).asInstanceOf[IfThen]
    val thenStore = branch.thenBlock.statements.head.asInstanceOf[Store[?, ?]]
    val elseStore =
      branch.elseBlock.get.statements.head.asInstanceOf[Store[?, ?]]
    val outputStore = statements(5).asInstanceOf[Store[?, ?]]
    val reduction = outputStore.value
      .asInstanceOf[Binary[Float]]
      .left
      .asInstanceOf[ReduceSum[Float, Float]]

    assertCaptured(
      Vector(coefficients.span) ++
        definition.ir.sharedMemory.map(_.span) ++
        Vector(
          localArrayDeclaration.span,
          localArrayDeclaration.array.span,
          localDeclaration.span,
          localDeclaration.local.span,
          firstStore.span,
          loop.span,
          loop.index.span,
          accumulation.span,
          branch.span,
          thenStore.span,
          elseStore.span,
          reduction.span,
          reduction.index.span,
          outputStore.span
        )
    )
