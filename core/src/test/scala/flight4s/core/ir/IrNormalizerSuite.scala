package flight4s.core.ir

import munit.FunSuite

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.types.*

class IrNormalizerSuite extends FunSuite:
  test("folds nested I32 arithmetic and preserves the outer source span"):
    val innerSpan = SourceSpan("Kernel.scala", 4, 12, 4, 17)
    val outerSpan = SourceSpan("Kernel.scala", 4, 11, 4, 22)
    val expression = Binary(
      BinaryOperator.Multiply,
      Binary(
        BinaryOperator.Add,
        Literal(2, I32, innerSpan),
        Literal(3, I32, innerSpan),
        I32,
        innerSpan
      ),
      Literal(4, I32, innerSpan),
      I32,
      outerSpan
    )

    val normalized = IrNormalizer.expression(expression)

    assertEquals(normalized, Literal(20, I32, outerSpan))
    assertEquals(expression.span, outerSpan)

  test("folds I32 comparisons after recursively normalizing their operands"):
    val span = SourceSpan("Kernel.scala", 8, 7, 8, 22)
    val comparison = Compare(
      ComparisonOperator.LessThan,
      Binary(
        BinaryOperator.Add,
        Literal(1, I32),
        Literal(2, I32),
        I32
      ),
      Literal(4, I32),
      I32,
      span
    )

    assertEquals(IrNormalizer.expression(comparison), Literal(true, Bool, span))

  test("does not fold zero divisors or signed I32 overflow"):
    val divisionByZero = Binary(
      BinaryOperator.Divide,
      Literal(1, I32),
      Literal(0, I32),
      I32
    )
    val overflow = Binary(
      BinaryOperator.Add,
      Literal(Int.MaxValue, I32),
      Literal(1, I32),
      I32
    )

    assertEquals(IrNormalizer.expression(divisionByZero), divisionByZero)
    assertEquals(IrNormalizer.expression(overflow), overflow)

  test("simplifies evaluation-preserving I32 identities and keeps the outer span"):
    val childSpan = SourceSpan("Kernel.scala", 5, 14, 5, 25)
    val outerSpan = SourceSpan("Kernel.scala", 5, 11, 5, 30)
    val value = Intrinsic("threadIdx.x", I32, childSpan)
    val expected = Intrinsic("threadIdx.x", I32, outerSpan)
    val expressions = Vector(
      Binary(BinaryOperator.Add, value, Literal(0, I32), I32, outerSpan),
      Binary(BinaryOperator.Add, Literal(0, I32), value, I32, outerSpan),
      Binary(BinaryOperator.Subtract, value, Literal(0, I32), I32, outerSpan),
      Binary(BinaryOperator.Multiply, value, Literal(1, I32), I32, outerSpan),
      Binary(BinaryOperator.Multiply, Literal(1, I32), value, I32, outerSpan),
      Binary(BinaryOperator.Divide, value, Literal(1, I32), I32, outerSpan)
    )

    expressions.foreach { expression =>
      assertEquals(IrNormalizer.expression(expression), expected)
    }

  test("does not apply evaluation-eliminating or non-I32 identities"):
    val input = Load(
      BufferElement[Int, ReadOnly]("input", Literal(0, I32), I32)
    )
    val multiplyByZero = Binary(
      BinaryOperator.Multiply,
      input,
      Literal(0, I32),
      I32
    )
    val remainderByOne = Binary(
      BinaryOperator.Remainder,
      input,
      Literal(1, I32),
      I32
    )
    val selfSubtract = Binary(BinaryOperator.Subtract, input, input, I32)
    val floatIdentity = Binary(
      BinaryOperator.Add,
      Intrinsic("value", F32),
      Literal(0.0f, F32),
      F32
    )

    assertEquals(IrNormalizer.expression(multiplyByZero), multiplyByZero)
    assertEquals(IrNormalizer.expression(remainderByOne), remainderByOne)
    assertEquals(IrNormalizer.expression(selfSubtract), selfSubtract)
    assertEquals(IrNormalizer.expression(floatIdentity), floatIdentity)

  test("normalizes kernel expression positions and is idempotent"):
    val result = output[Int]("result")
    val definition = kernel("foldKernel", params(result)) { bindings =>
      val index = literal(1) + literal(2)
      bindings.head(index) := (literal(3) * literal(4))
    }

    val normalized = IrNormalizer.kernel(definition.ir)
    val store = normalized.body.statements.head.asInstanceOf[Store[Int, Global]]
    val target = store.to.asInstanceOf[BufferElement[Int, ReadWrite]]

    assertEquals(target.index, Literal(3, I32))
    assertEquals(store.value, Literal(12, I32))
    assertEquals(normalized.signature, definition.ir.signature)
    assertEquals(IrNormalizer.kernel(normalized), normalized)

  test("propagates normalized I32 locals through later reads"):
    val width = LocalVariable("width", I32)
    val result = output[Int]("result")
    val definition = KernelIR(
      "propagateLocal",
      params(result),
      Block(
        Vector(
          LocalDeclaration(
            width,
            Binary(
              BinaryOperator.Add,
              Literal(2, I32),
              Literal(3, I32),
              I32
            )
          ),
          Store(
            BufferElement[Int, ReadWrite]("result", Load(width), I32),
            Binary(
              BinaryOperator.Multiply,
              Load(width),
              Literal(4, I32),
              I32
            )
          )
        )
      )
    )

    val normalized = IrNormalizer.kernel(definition)
    val declaration = normalized.body.statements.head.asInstanceOf[LocalDeclaration[Int]]
    val store = normalized.body.statements(1).asInstanceOf[Store[Int, Global]]
    val target = store.to.asInstanceOf[BufferElement[Int, ReadWrite]]

    assertEquals(declaration.initial, Literal(5, I32))
    assertEquals(target.index, Literal(5, I32))
    assertEquals(store.value, Literal(20, I32))

  test("updates direct-store facts and clears facts after accumulation"):
    val value = LocalVariable("value", I32)
    val result = BufferElement[Int, ReadWrite]("result", literal(0), I32)
    val directStore = KernelIR(
      "directStore",
      params(output[Int]("result")),
      Block(
        Vector(
          LocalDeclaration(value, Literal(1, I32)),
          Store(value, Literal(2, I32)),
          Store(result, Load(value))
        )
      )
    )
    val accumulation = KernelIR(
      "accumulation",
      params(output[Int]("result")),
      Block(
        Vector(
          LocalDeclaration(value, Literal(1, I32)),
          Accumulate(value, Literal(2, I32), I32),
          Store(result, Load(value))
        )
      )
    )

    val directStoreResult = IrNormalizer.kernel(directStore)
      .body
      .statements(2)
      .asInstanceOf[Store[Int, Global]]
    val accumulationResult = IrNormalizer.kernel(accumulation)
      .body
      .statements(2)
      .asInstanceOf[Store[Int, Global]]

    assertEquals(directStoreResult.value, Literal(2, I32))
    assertEquals(accumulationResult.value, Load(value))

  test("prunes literal branches and retains the selected lexical scope"):
    val firstSpan = SourceSpan("Kernel.scala", 8, 5, 14, 6)
    val secondSpan = SourceSpan("Kernel.scala", 16, 5, 20, 6)
    val temporary = LocalVariable("temporary", I32)
    val firstResult = BufferElement[Int, ReadWrite]("result", literal(0), I32)
    val secondResult = BufferElement[Int, ReadWrite]("result", literal(1), I32)
    val definition = KernelIR(
      "pruneBranches",
      params(output[Int]("result")),
      Block(
        Vector(
          IfThen(
            Literal(true, Bool),
            Block(
              Vector(
                LocalDeclaration(temporary, Literal(1, I32)),
                Store(firstResult, Load(temporary))
              )
            ),
            Some(Block(Vector(Store(firstResult, Literal(99, I32))))),
            firstSpan
          ),
          IfThen(
            Literal(false, Bool),
            Block(Vector(Store(secondResult, Literal(99, I32)))),
            Some(Block(Vector(Store(secondResult, Literal(2, I32))))),
            secondSpan
          )
        )
      )
    )

    val normalized = IrNormalizer.kernel(definition)
    val first = normalized.body.statements.head.asInstanceOf[ScopedBlock]
    val second = normalized.body.statements(1).asInstanceOf[ScopedBlock]
    val firstStore = first.body.statements(1).asInstanceOf[Store[Int, Global]]
    val secondStore = second.body.statements.head.asInstanceOf[Store[Int, Global]]

    assertEquals(first.span, firstSpan)
    assertEquals(second.span, secondSpan)
    assertEquals(firstStore.value, Literal(1, I32))
    assertEquals(secondStore.value, Literal(2, I32))
    assertEquals(IrNormalizer.kernel(normalized), normalized)

  test("removes no-op literal branches without clearing straight-line facts"):
    val value = LocalVariable("value", I32)
    val result = BufferElement[Int, ReadWrite]("result", literal(0), I32)
    val definition = KernelIR(
      "removeBranches",
      params(output[Int]("result")),
      Block(
        Vector(
          LocalDeclaration(value, Literal(5, I32)),
          IfThen(
            Literal(false, Bool),
            Block(Vector(Store(value, Literal(9, I32))))
          ),
          IfThen(Literal(true, Bool), Block(Vector.empty)),
          Store(result, Load(value))
        )
      )
    )

    val normalized = IrNormalizer.kernel(definition)
    val store = normalized.body.statements(1).asInstanceOf[Store[Int, Global]]

    assertEquals(normalized.body.statements.size, 2)
    assertEquals(store.value, Literal(5, I32))

  test("eliminates repeated stable I32 expressions within one statement"):
    val firstSpan = SourceSpan("Kernel.scala", 7, 15, 7, 30)
    val secondSpan = SourceSpan("Kernel.scala", 7, 33, 7, 48)
    val first = Binary(
      BinaryOperator.Add,
      Intrinsic("threadIdx.x", I32),
      Literal(2, I32),
      I32,
      firstSpan
    )
    val second = first.copy(span = secondSpan)
    val result = BufferElement[Int, ReadWrite]("result", literal(0), I32)
    val occupied = LocalVariable("flight4s_cse_0", I32)
    val definition = KernelIR(
      "localCse",
      params(output[Int]("result")),
      Block(
        Vector(
          LocalDeclaration(occupied, Literal(7, I32)),
          Store(
            result,
            Binary(BinaryOperator.Multiply, first, second, I32)
          )
        )
      )
    )

    val normalized = IrNormalizer.kernel(definition)
    val generated = normalized.body.statements(1).asInstanceOf[LocalDeclaration[Int]]
    val store = normalized.body.statements(2).asInstanceOf[Store[Int, Global]]
    val product = store.value.asInstanceOf[Binary[Int]]

    assertEquals(generated.local.name, "flight4s_cse_1")
    assertEquals(generated.initial, first)
    assertEquals(product.left, Load(generated.local, firstSpan))
    assertEquals(product.right, Load(generated.local, secondSpan))
    assertEquals(IrNormalizer.kernel(normalized), normalized)

  test("keeps CSE within stable I32 expressions and one statement"):
    val inputElement = BufferElement[Int, ReadOnly]("input", literal(0), I32)
    val loaded = Binary(
      BinaryOperator.Add,
      Load(inputElement),
      Literal(1, I32),
      I32
    )
    val floatValue = Binary(
      BinaryOperator.Add,
      Intrinsic("value", F32),
      Literal(1.0f, F32),
      F32
    )
    val intOutput = BufferElement[Int, ReadWrite]("intOutput", literal(0), I32)
    val floatOutput = BufferElement[Float, ReadWrite]("floatOutput", literal(0), F32)
    val repeated = Binary(
      BinaryOperator.Add,
      Intrinsic("threadIdx.x", I32),
      Literal(2, I32),
      I32
    )
    val definition = KernelIR(
      "cseBoundaries",
      params(
        input[Int]("input"),
        output[Int]("intOutput"),
        output[Float]("floatOutput")
      ),
      Block(
        Vector(
          Store(
            intOutput,
            Binary(BinaryOperator.Add, loaded, loaded, I32)
          ),
          Store(intOutput, repeated),
          Store(intOutput, repeated),
          Store(
            floatOutput,
            Binary(BinaryOperator.Multiply, floatValue, floatValue, F32)
          )
        )
      )
    )

    val normalized = IrNormalizer.kernel(definition)

    assertEquals(normalized.body.statements.size, definition.body.statements.size)
    assertEquals(normalized.body, definition.body)

  test("does not hoist repeated expressions out of reduction scope"):
    val result = output[Int]("result")
    val definition = kernel("reductionCseBoundary", params(result)) { bindings =>
      val sum = reduceSum("index", literal(0), literal(4), literal(0)) { index =>
        val repeated = index + literal(1)
        repeated * repeated
      }
      bindings.head(literal(0)) := sum
    }

    val normalized = IrNormalizer.kernel(definition.ir)
    val store = normalized.body.statements.head.asInstanceOf[Store[Int, Global]]
    val reduction = store.value.asInstanceOf[ReduceSum[Int, Int]]

    assertEquals(normalized.body.statements.size, 1)
    assert(reduction.value.isInstanceOf[Binary[?]])

  test("does not propagate outer local facts through structured control flow"):
    val value = LocalVariable("value", I32)
    val index = LoopIndex("index")
    val result = BufferElement[Int, ReadWrite]("result", literal(0), I32)
    val definition = KernelIR(
      "controlFlowBoundary",
      params(output[Int]("result")),
      Block(
        Vector(
          LocalDeclaration(value, Literal(5, I32)),
          IfThen(threadIdx.x < literal(1), Block(Vector.empty)),
          Store(result, Load(value)),
          ForLoop(
            index,
            Literal(0, I32),
            Literal(2, I32),
            Block(Vector(Store(result, Load(value))))
          )
        )
      )
    )

    val normalized = IrNormalizer.kernel(definition)
    val afterBranch = normalized.body.statements(2).asInstanceOf[Store[Int, Global]]
    val loop = normalized.body.statements(3).asInstanceOf[ForLoop]
    val insideLoop = loop.body.statements.head.asInstanceOf[Store[Int, Global]]

    assertEquals(afterBranch.value, Load(value))
    assertEquals(insideLoop.value, Load(value))

  test("normalization preserves observable effects and their order"):
    def observableTrace(block: Block): Vector[String] =
      block.statements.collect:
        case store: Store[?, ?] =>
          store.to match
            case buffer: BufferElement[?, ?] =>
              s"store:global:${buffer.bufferName}"
            case shared: SharedElement[?] =>
              s"store:shared:${shared.arrayName}"
            case local: LocalVariable[?] =>
              s"store:local:${local.name}"
            case localArray: LocalArrayElement[?] =>
              s"store:local-array:${localArray.arrayName}"
        case accumulation: Accumulate[?] =>
          s"accumulate:${accumulation.target.name}"
        case _: Barrier => "barrier"

    val accumulator = LocalVariable("accumulator", I32)
    val repeatedLeft = Binary(
      BinaryOperator.Add,
      Intrinsic("threadIdx.x", I32),
      Literal(1, I32),
      I32
    )
    val repeatedRight = repeatedLeft.copy()
    val original = Block(
      Vector(
        LocalDeclaration(
          accumulator,
          Binary(
            BinaryOperator.Add,
            Literal(1, I32),
            Literal(2, I32),
            I32
          )
        ),
        Store(
          BufferElement[Int, ReadWrite]("output", Literal(0, I32), I32),
          Binary(
            BinaryOperator.Multiply,
            repeatedLeft,
            repeatedRight,
            I32
          )
        ),
        Store(
          SharedElement("tile", Vector(Literal(0, I32)), I32),
          Literal(7, I32)
        ),
        Barrier(),
        Accumulate(
          accumulator,
          Literal(4, I32),
          summon[AdditiveType[Int]]
        ),
        Store(
          BufferElement[Int, ReadWrite]("output", Literal(1, I32), I32),
          Load(accumulator)
        ),
        Barrier()
      )
    )

    val normalized = IrNormalizer.block(original)
    val normalizedAccumulator =
      normalized.statements.head.asInstanceOf[LocalDeclaration[Int]]

    assertEquals(normalizedAccumulator.initial, Literal(3, I32))
    assertEquals(normalized.statements.size, original.statements.size + 1)
    assertEquals(EffectAnalysis.block(normalized), EffectAnalysis.block(original))
    assertEquals(
      observableTrace(original),
      Vector(
        "store:global:output",
        "store:shared:tile",
        "barrier",
        "accumulate:accumulator",
        "store:global:output",
        "barrier"
      )
    )
    assertEquals(observableTrace(normalized), observableTrace(original))
    assertEquals(IrNormalizer.block(normalized), normalized)
