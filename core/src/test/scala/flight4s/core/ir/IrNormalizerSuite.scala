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
