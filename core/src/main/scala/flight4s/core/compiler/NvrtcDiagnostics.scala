package flight4s.core.compiler

import flight4s.core.codegen.SourceMap
import flight4s.core.ir.SourceSpan

enum NvrtcDiagnosticSeverity(val label: String):
  case Error extends NvrtcDiagnosticSeverity("error")
  case Warning extends NvrtcDiagnosticSeverity("warning")
  case Note extends NvrtcDiagnosticSeverity("note")
  case Remark extends NvrtcDiagnosticSeverity("remark")

final case class NvrtcGeneratedLocation(
    file: String,
    line: Int,
    column: Option[Int]
):
  require(file.nonEmpty, "generated diagnostic file must not be empty")
  require(line > 0, "generated diagnostic line must be positive")
  require(
    column.forall(_ > 0),
    "generated diagnostic column must be positive"
  )

  def render: String =
    column match
      case Some(value) => s"$file:$line:$value"
      case None => s"$file:$line"

final case class NvrtcDiagnostic(
    severity: NvrtcDiagnosticSeverity,
    code: Option[String],
    message: String,
    generatedLocation: NvrtcGeneratedLocation,
    sourceSpan: Option[SourceSpan],
    rawLine: String
):
  require(code.forall(_.nonEmpty), "diagnostic code must not be empty")

  def render: String =
    val renderedCode = code.fold("")(value => s" $value")
    val generated =
      s"${generatedLocation.render}: ${severity.label}$renderedCode: $message"
    sourceSpan match
      case Some(span) =>
        generated +
          s"\nScala source: ${span.file}:${span.startLine}:${span.startColumn}"
      case None => generated

object NvrtcDiagnostics:
  private val ParenthesizedLocation =
    """^(.+?)\((\d+)(?:,(\d+))?\):\s*(catastrophic error|error|warning|note|remark)(?:\s+(#[^:]+))?\s*:\s*(.*)$""".r

  private val ColonLocation =
    """^(.+?):(\d+)(?::(\d+))?:\s*(catastrophic error|error|warning|note|remark)(?:\s+(#[^:]+))?\s*:\s*(.*)$""".r

  def parse(
      compileLog: String,
      sourceMap: SourceMap,
      generatedFile: String
  ): Vector[NvrtcDiagnostic] =
    require(generatedFile.nonEmpty, "generated file must not be empty")
    compileLog.linesIterator.flatMap { rawLine =>
      rawLine match
        case ParenthesizedLocation(
              file,
              line,
              column,
              severity,
              code,
              message
            ) =>
          build(
            file,
            line,
            column,
            severity,
            code,
            message,
            rawLine,
            sourceMap,
            generatedFile
          )
        case ColonLocation(
              file,
              line,
              column,
              severity,
              code,
              message
            ) =>
          build(
            file,
            line,
            column,
            severity,
            code,
            message,
            rawLine,
            sourceMap,
            generatedFile
          )
        case _ => None
    }.toVector

  private def build(
      file: String,
      line: String,
      column: String,
      severity: String,
      code: String,
      message: String,
      rawLine: String,
      sourceMap: SourceMap,
      generatedFile: String
  ): Option[NvrtcDiagnostic] =
    for
      generatedLine <- line.toIntOption.filter(_ > 0)
      generatedColumn <- parseColumn(column)
    yield
      NvrtcDiagnostic(
        severity = parseSeverity(severity),
        code = Option(code).map(_.trim).filter(_.nonEmpty),
        message = message.trim,
        generatedLocation = NvrtcGeneratedLocation(
          file = file.trim,
          line = generatedLine,
          column = generatedColumn
        ),
        sourceSpan =
          if sameFile(file.trim, generatedFile) then
            sourceMap
              .closestAtOrBefore(generatedLine)
              .map(_.sourceSpan)
              .filter(_ != SourceSpan.Unknown)
          else None,
        rawLine = rawLine
      )

  private def parseColumn(value: String): Option[Option[Int]] =
    Option(value) match
      case Some(raw) =>
        raw.toIntOption.filter(_ > 0).map(Some(_))
      case None => Some(None)

  private def parseSeverity(value: String): NvrtcDiagnosticSeverity =
    value match
      case "catastrophic error" | "error" =>
        NvrtcDiagnosticSeverity.Error
      case "warning" => NvrtcDiagnosticSeverity.Warning
      case "note" => NvrtcDiagnosticSeverity.Note
      case "remark" => NvrtcDiagnosticSeverity.Remark
      case other =>
        throw IllegalArgumentException(s"unsupported NVRTC severity: $other")

  private def sameFile(reported: String, generated: String): Boolean =
    def fileName(value: String): String =
      value.replace('\\', '/').split('/').lastOption.getOrElse(value)

    reported == generated || fileName(reported) == fileName(generated)
