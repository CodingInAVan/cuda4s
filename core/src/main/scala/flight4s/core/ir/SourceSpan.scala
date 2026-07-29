package flight4s.core.ir

final case class SourceSpan(
    file: String,
    startLine: Int,
    startColumn: Int,
    endLine: Int,
    endColumn: Int
)

object SourceSpan:
  val Unknown: SourceSpan = SourceSpan(
    file = "<unknown>",
    startLine = 0,
    startColumn = 0,
    endLine = 0,
    endColumn = 0
  )
