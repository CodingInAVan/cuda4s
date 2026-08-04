package flight4s.core.ir

private[core] enum Uniformity:
  case GridUniform
  case BlockUniform
  case WarpUniform
  case Varying
  case Unknown

  def join(other: Uniformity): Uniformity = (this, other) match
    case (Uniformity.Unknown, _) | (_, Uniformity.Unknown) => Uniformity.Unknown
    case (Uniformity.Varying, _) | (_, Uniformity.Varying) => Uniformity.Varying
    case (Uniformity.WarpUniform, _) | (_, Uniformity.WarpUniform) =>
      Uniformity.WarpUniform
    case (Uniformity.BlockUniform, _) | (_, Uniformity.BlockUniform) =>
      Uniformity.BlockUniform
    case _ => Uniformity.GridUniform

private[core] object UniformityAnalysis:
  private val intrinsicUniformities: Map[String, Uniformity] = Map(
    "threadIdx.x" -> Uniformity.Varying,
    "threadIdx.y" -> Uniformity.Varying,
    "threadIdx.z" -> Uniformity.Varying,
    "blockIdx.x" -> Uniformity.BlockUniform,
    "blockIdx.y" -> Uniformity.BlockUniform,
    "blockIdx.z" -> Uniformity.BlockUniform,
    "blockDim.x" -> Uniformity.GridUniform,
    "blockDim.y" -> Uniformity.GridUniform,
    "blockDim.z" -> Uniformity.GridUniform
  )

  def expression(expr: Expr[?]): Uniformity = expr match
    case _: Literal[?] => Uniformity.GridUniform
    case binary: Binary[?] =>
      expression(binary.left).join(expression(binary.right))
    case comparison: Compare[?] =>
      expression(comparison.left).join(expression(comparison.right))
    case intrinsic: Intrinsic[?] =>
      intrinsicUniformities.getOrElse(intrinsic.name, Uniformity.Unknown)
    case conversion: Convert[?, ?] => expression(conversion.value)
    case accumulation: ToAccumulator[?, ?] => expression(accumulation.value)
    case _: ReductionIndex => Uniformity.Unknown
    case _: LoopIndex => Uniformity.Unknown
    case reduction: ReduceSum[?, ?] =>
      expression(reduction.from)
        .join(expression(reduction.until))
        .join(expression(reduction.initial))
        .join(expression(reduction.value))
    case load: Load[?, ?, ?] => loadUniformity(load.from)
    case _: ScalarParam[?] => Uniformity.GridUniform

  private def loadUniformity(place: Place[?, ?, ?]): Uniformity = place match
    case constant: ConstantElement[?] => expression(constant.index)
    case _: LocalVariable[?] => Uniformity.Unknown
    case _: BufferElement[?, ?] | _: SharedElement[?] | _: LocalArrayElement[?] =>
      Uniformity.Varying
