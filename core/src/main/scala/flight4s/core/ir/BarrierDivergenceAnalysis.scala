package flight4s.core.ir

private[core] object BarrierDivergenceAnalysis:
  def warnings(block: Block): Vector[ValidationWarning] =
    blockWarnings(block, "body", controlMayDiverge = false)

  private def blockWarnings(
      block: Block,
      location: String,
      controlMayDiverge: Boolean
  ): Vector[ValidationWarning] =
    block.statements.zipWithIndex.flatMap { case (statement, index) =>
      val statementLocation = s"$location.statements[$index]"
      statement match
        case branch: IfThen =>
          val branchMayDiverge =
            controlMayDiverge || isKnownDivergent(branch.condition)
          blockWarnings(branch.thenBlock, s"$statementLocation.then", branchMayDiverge) ++
            branch.elseBlock.toVector.flatMap(
              blockWarnings(_, s"$statementLocation.else", branchMayDiverge)
            )

        case loop: ForLoop =>
          val loopMayDiverge =
            controlMayDiverge ||
              isKnownDivergent(loop.from) ||
              isKnownDivergent(loop.until)
          blockWarnings(loop.body, s"$statementLocation.body", loopMayDiverge)

        case barrier: Barrier if controlMayDiverge =>
          Vector(
            ValidationWarning(
              ValidationWarningCode.BarrierMayDiverge,
              "block barrier may be reached through divergent control flow",
              statementLocation,
              barrier.span
            )
          )

        case _: Barrier => Vector.empty
        case _: LocalDeclaration[?] => Vector.empty
        case _: LocalArrayDeclaration[?] => Vector.empty
        case _: Store[?, ?] => Vector.empty
        case _: Accumulate[?] => Vector.empty
    }

  private def isKnownDivergent(expression: Expr[?]): Boolean =
    UniformityAnalysis.expression(expression) match
      case Uniformity.WarpUniform | Uniformity.Varying => true
      case Uniformity.GridUniform | Uniformity.BlockUniform | Uniformity.Unknown =>
        false
