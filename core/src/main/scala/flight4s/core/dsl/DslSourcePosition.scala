package flight4s.core.dsl

import scala.quoted.*

import flight4s.core.ir.SourceSpan

final case class DslSourcePosition(span: SourceSpan)

object DslSourcePosition:
  inline given generated: DslSourcePosition = ${ capture }

  private def capture(using quotes: Quotes): Expr[DslSourcePosition] =
    import quotes.reflect.*

    val position = Position.ofMacroExpansion
    val span = SourceSpan(
      file = position.sourceFile.path,
      startLine = position.startLine + 1,
      startColumn = position.startColumn + 1,
      endLine = position.endLine + 1,
      endColumn = position.endColumn + 1
    )
    '{ DslSourcePosition(${ Expr(span) }) }

  private given ToExpr[SourceSpan] with
    def apply(span: SourceSpan)(using Quotes): Expr[SourceSpan] =
      '{
        SourceSpan(
          file = ${ Expr(span.file) },
          startLine = ${ Expr(span.startLine) },
          startColumn = ${ Expr(span.startColumn) },
          endLine = ${ Expr(span.endLine) },
          endColumn = ${ Expr(span.endColumn) }
        )
      }
