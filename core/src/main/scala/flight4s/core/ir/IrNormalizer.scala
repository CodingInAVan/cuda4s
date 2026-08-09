package flight4s.core.ir

import flight4s.core.types.{Bool, I32}

private[core] object IrNormalizer:
  def module(module: CudaModuleIR): CudaModuleIR =
    module.copy(kernels = module.kernels.map(kernel))

  def kernel[Args <: Tuple](kernel: KernelIR[Args]): KernelIR[Args] =
    kernel.copy(body = block(kernel.body))

  def block(block: Block): Block =
    Block(block.statements.map(statement))

  def statement(statement: Stmt): Stmt = statement match
    case declaration: LocalDeclaration[?] =>
      normalizeLocalDeclaration(declaration)
    case declaration: LocalArrayDeclaration[?] => declaration
    case store: Store[?, ?] => normalizeStore(store)
    case accumulation: Accumulate[?] =>
      accumulation.copy(value = expression(accumulation.value))
    case branch: IfThen =>
      branch.copy(
        condition = expression(branch.condition),
        thenBlock = block(branch.thenBlock),
        elseBlock = branch.elseBlock.map(block)
      )
    case loop: ForLoop =>
      loop.copy(
        from = expression(loop.from),
        until = expression(loop.until),
        body = block(loop.body)
      )
    case barrier: Barrier => barrier

  def expression[T](expr: Expr[T]): Expr[T] = expr match
    case literal: Literal[?] => literal.asInstanceOf[Expr[T]]
    case binary: Binary[?] =>
      normalizeBinary(binary).asInstanceOf[Expr[T]]
    case comparison: Compare[?] =>
      normalizeComparison(comparison).asInstanceOf[Expr[T]]
    case intrinsic: Intrinsic[?] => intrinsic.asInstanceOf[Expr[T]]
    case conversion: Convert[?, ?] =>
      conversion
        .copy(value = expression(conversion.value))
        .asInstanceOf[Expr[T]]
    case accumulation: ToAccumulator[?, ?] =>
      accumulation
        .copy(value = expression(accumulation.value))
        .asInstanceOf[Expr[T]]
    case index: ReductionIndex => index.asInstanceOf[Expr[T]]
    case index: LoopIndex => index.asInstanceOf[Expr[T]]
    case reduction: ReduceSum[?, ?] =>
      reduction
        .copy(
          from = expression(reduction.from),
          until = expression(reduction.until),
          initial = expression(reduction.initial),
          value = expression(reduction.value)
        )
        .asInstanceOf[Expr[T]]
    case load: Load[?, ?, ?] =>
      load.copy(from = place(load.from)).asInstanceOf[Expr[T]]
    case parameter: ScalarParam[?] => parameter.asInstanceOf[Expr[T]]

  private def normalizeLocalDeclaration[T](
      declaration: LocalDeclaration[T]
  ): LocalDeclaration[T] =
    declaration.copy(initial = expression(declaration.initial))

  private def normalizeStore[T, Space <: AddressSpace](
      store: Store[T, Space]
  ): Store[T, Space] =
    store.copy(
      to = place(store.to),
      value = expression(store.value)
    )

  private def normalizeBinary[T](binary: Binary[T]): Expr[T] =
    val left = expression(binary.left)
    val right = expression(binary.right)
    foldIntegerBinary(binary, left, right).getOrElse(
      binary.copy(left = left, right = right)
    )

  private def normalizeComparison[T](comparison: Compare[T]): Expr[Boolean] =
    val left = expression(comparison.left)
    val right = expression(comparison.right)
    foldIntegerComparison(comparison, left, right).getOrElse(
      comparison.copy(left = left, right = right)
    )

  private def place[
      T,
      Space <: AddressSpace,
      Mode <: AccessMode
  ](place: Place[T, Space, Mode]): Place[T, Space, Mode] = place match
    case buffer: BufferElement[?, ?] =>
      buffer.copy(index = expression(buffer.index)).asInstanceOf[Place[T, Space, Mode]]
    case constant: ConstantElement[?] =>
      constant
        .copy(index = expression(constant.index))
        .asInstanceOf[Place[T, Space, Mode]]
    case shared: SharedElement[?] =>
      shared
        .copy(indices = shared.indices.map(expression))
        .asInstanceOf[Place[T, Space, Mode]]
    case local: LocalVariable[?] => local.asInstanceOf[Place[T, Space, Mode]]
    case localArray: LocalArrayElement[?] =>
      localArray
        .copy(index = expression(localArray.index))
        .asInstanceOf[Place[T, Space, Mode]]

  private def foldIntegerBinary[T](
      binary: Binary[T],
      left: Expr[T],
      right: Expr[T]
  ): Option[Expr[T]] =
    if binary.valueType != I32 then None
    else
      (left, right) match
        case (leftLiteral: Literal[?], rightLiteral: Literal[?]) =>
          (leftLiteral.value, rightLiteral.value) match
            case (leftValue: Int, rightValue: Int) =>
              integerBinaryResult(binary.operator, leftValue, rightValue).map { value =>
                Literal(value, I32, binary.span).asInstanceOf[Expr[T]]
              }
            case _ => None
        case _ => None

  private def integerBinaryResult(
      operator: BinaryOperator,
      left: Int,
      right: Int
  ): Option[Int] =
    operator match
      case BinaryOperator.Add =>
        try Some(Math.addExact(left, right))
        catch case _: ArithmeticException => None
      case BinaryOperator.Subtract =>
        try Some(Math.subtractExact(left, right))
        catch case _: ArithmeticException => None
      case BinaryOperator.Multiply =>
        try Some(Math.multiplyExact(left, right))
        catch case _: ArithmeticException => None
      case BinaryOperator.Divide if right == 0 || (left == Int.MinValue && right == -1) =>
        None
      case BinaryOperator.Divide => Some(left / right)
      case BinaryOperator.Remainder
          if right == 0 || (left == Int.MinValue && right == -1) =>
        None
      case BinaryOperator.Remainder => Some(left % right)

  private def foldIntegerComparison[T](
      comparison: Compare[T],
      left: Expr[T],
      right: Expr[T]
  ): Option[Expr[Boolean]] =
    if comparison.operandType != I32 then None
    else
      (left, right) match
        case (leftLiteral: Literal[?], rightLiteral: Literal[?]) =>
          (leftLiteral.value, rightLiteral.value) match
            case (leftValue: Int, rightValue: Int) =>
              Some(
                Literal(
                  integerComparisonResult(comparison.operator, leftValue, rightValue),
                  Bool,
                  comparison.span
                )
              )
            case _ => None
        case _ => None

  private def integerComparisonResult(
      operator: ComparisonOperator,
      left: Int,
      right: Int
  ): Boolean =
    operator match
      case ComparisonOperator.LessThan => left < right
      case ComparisonOperator.LessThanOrEqual => left <= right
      case ComparisonOperator.GreaterThan => left > right
      case ComparisonOperator.GreaterThanOrEqual => left >= right
      case ComparisonOperator.Equal => left == right
      case ComparisonOperator.NotEqual => left != right
