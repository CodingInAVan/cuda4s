package com.cuda4s.core.ir

import com.cuda4s.core.types.{Bool, CudaType, I32}

enum ValidationCode:
  case InvalidKernelName
  case InvalidParameterName
  case DuplicateParameterName
  case InvalidLocalName
  case DuplicateLocalName
  case LocalNameConflictsWithBinding
  case UnboundLocal
  case LocalTypeMismatch
  case InvalidLoopIndexName
  case LoopIndexConflictsWithBinding
  case UnboundLoopIndex
  case UnknownBuffer
  case ExpectedBuffer
  case BufferTypeMismatch
  case WriteToReadOnlyBuffer
  case ExpressionTypeMismatch
  case UnknownIntrinsic
  case InvalidReductionIndexName
  case DuplicateReductionIndex
  case ReductionIndexConflictsWithParameter
  case UnboundReductionIndex

final case class ValidationError(
    code: ValidationCode,
    message: String,
    location: String,
    span: SourceSpan = SourceSpan.Unknown
)

final case class ValidationResult(errors: Vector[ValidationError]):
  def isValid: Boolean = errors.isEmpty

  def toEither: Either[Vector[ValidationError], Unit] =
    if isValid then Right(()) else Left(errors)

object KernelValidator:
  private final case class ValidationScope(
      locals: Map[String, CudaType[?]] = Map.empty,
      loopIndexes: Set[String] = Set.empty,
      reductionIndexes: Set[String] = Set.empty
  )

  private val cudaIdentifier = raw"[A-Za-z_][A-Za-z0-9_]*".r
  private val intrinsicTypes: Map[String, CudaType[?]] = Map(
    "threadIdx.x" -> I32,
    "threadIdx.y" -> I32,
    "threadIdx.z" -> I32,
    "blockIdx.x" -> I32,
    "blockIdx.y" -> I32,
    "blockIdx.z" -> I32,
    "blockDim.x" -> I32,
    "blockDim.y" -> I32,
    "blockDim.z" -> I32
  )

  def validate(kernel: KernelIR): ValidationResult =
    val parameterErrors = validateParameters(kernel)
    val parametersByName = kernel.params.groupBy(_.name).view.mapValues(_.head).toMap
    val bodyErrors = validateBlock(kernel.body, parametersByName, "body")

    ValidationResult(parameterErrors ++ bodyErrors)

  private def validateParameters(kernel: KernelIR): Vector[ValidationError] =
    val kernelNameErrors =
      if isIdentifier(kernel.name) then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.InvalidKernelName,
            s"'${kernel.name}' is not a valid CUDA kernel identifier",
            "kernel"
          )
        )

    val nameErrors = kernel.params.zipWithIndex.flatMap { case (parameter, index) =>
      if isIdentifier(parameter.name) then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.InvalidParameterName,
            s"'${parameter.name}' is not a valid CUDA parameter identifier",
            s"params[$index]"
          )
        )
    }

    val duplicateErrors = kernel.params
      .groupBy(_.name)
      .collect { case (name, parameters) if parameters.size > 1 =>
        ValidationError(
          ValidationCode.DuplicateParameterName,
          s"CUDA parameter '$name' is declared ${parameters.size} times",
          "params"
        )
      }
      .toVector
      .sortBy(_.message)

    kernelNameErrors ++ nameErrors ++ duplicateErrors

  private def validateBlock(
      block: Block,
      parameters: Map[String, KernelParam],
      location: String,
      initialScope: ValidationScope = ValidationScope()
  ): Vector[ValidationError] =
    block.statements.zipWithIndex
      .foldLeft((Vector.empty[ValidationError], initialScope)) {
        case ((errors, scope), (declaration: LocalDeclaration[?], index)) =>
          val statementLocation = s"$location.statements[$index]"
          val nameErrors =
            validateLocalName(declaration.local, parameters, scope, statementLocation)
          val declarationErrors =
            nameErrors ++
              validateExpression(
                declaration.initial,
                parameters,
                s"$statementLocation.initial",
                scope
              ) ++
              requireSameType(
                declaration.initial.valueType,
                declaration.local.valueType,
                "local initializer type does not match the declared local type",
                s"$statementLocation.initial",
                declaration.initial.span,
                ValidationCode.LocalTypeMismatch
              )
          val nextScope =
            if nameErrors.isEmpty then
              scope.copy(
                locals = scope.locals.updated(
                  declaration.local.name,
                  declaration.local.valueType
                )
              )
            else scope

          (errors ++ declarationErrors, nextScope)

        case ((errors, scope), (statement, index)) =>
          (
            errors ++ validateStatement(
              statement,
              parameters,
              s"$location.statements[$index]",
              scope
            ),
            scope
          )
      }
      ._1

  private def validateStatement(
      statement: Stmt,
      parameters: Map[String, KernelParam],
      location: String,
      scope: ValidationScope
  ): Vector[ValidationError] =
    statement match
      case declaration: LocalDeclaration[?] =>
        validateLocalName(declaration.local, parameters, scope, location) ++
          validateExpression(
            declaration.initial,
            parameters,
            s"$location.initial",
            scope
          )

      case store: Store[?, ?] =>
        validatePlace(
          store.to,
          parameters,
          s"$location.to",
          isWrite = true,
          scope = scope
        ) ++
          validateExpression(store.value, parameters, s"$location.value", scope) ++
          requireSameType(
            store.to.valueType,
            store.value.valueType,
            s"store target type ${store.to.valueType.cudaName} does not match " +
              s"value type ${store.value.valueType.cudaName}",
            location,
            store.span
          )

      case accumulation: Accumulate[?] =>
        validatePlace(
          accumulation.target,
          parameters,
          s"$location.target",
          isWrite = true,
          scope = scope
        ) ++
          validateExpression(
            accumulation.value,
            parameters,
            s"$location.value",
            scope
          ) ++
          requireSameType(
            accumulation.target.valueType,
            accumulation.value.valueType,
            "accumulated value type does not match the local accumulator type",
            s"$location.value",
            accumulation.value.span
          ) ++
          requireSameType(
            accumulation.addition,
            accumulation.target.valueType,
            "accumulation capability does not match the local accumulator type",
            location,
            accumulation.span
          )

      case branch: IfThen =>
        validateExpression(
          branch.condition,
          parameters,
          s"$location.condition",
          scope
        ) ++
          requireSameType(
            branch.condition.valueType,
            Bool,
            "GPU branch condition must have CUDA bool type",
            s"$location.condition",
            branch.condition.span
          ) ++
          validateBlock(branch.thenBlock, parameters, s"$location.then", scope) ++
          branch.elseBlock.toVector.flatMap(
            validateBlock(_, parameters, s"$location.else", scope)
          )

      case loop: ForLoop =>
        val nameErrors =
          validateLoopIndexName(loop.index, parameters, scope, s"$location.index")
        val rangeErrors =
          validateExpression(loop.from, parameters, s"$location.from", scope) ++
            validateExpression(loop.until, parameters, s"$location.until", scope) ++
            requireSameType(
              loop.from.valueType,
              I32,
              "loop lower bound must have CUDA int type",
              s"$location.from",
              loop.from.span
            ) ++
            requireSameType(
              loop.until.valueType,
              I32,
              "loop upper bound must have CUDA int type",
              s"$location.until",
              loop.until.span
            )
        val bodyScope =
          scope.copy(loopIndexes = scope.loopIndexes + loop.index.name)

        nameErrors ++ rangeErrors ++
          validateBlock(loop.body, parameters, s"$location.body", bodyScope)

      case _: Barrier =>
        Vector.empty

  private def validateExpression(
      expression: Expr[?],
      parameters: Map[String, KernelParam],
      location: String,
      scope: ValidationScope = ValidationScope()
  ): Vector[ValidationError] =
    expression match
      case _: Literal[?] =>
        Vector.empty

      case binary: Binary[?] =>
        validateExpression(binary.left, parameters, s"$location.left", scope) ++
          validateExpression(binary.right, parameters, s"$location.right", scope) ++
          requireSameType(
            binary.left.valueType,
            binary.valueType,
            "left operand type does not match binary result type",
            s"$location.left",
            binary.left.span
          ) ++
          requireSameType(
            binary.right.valueType,
            binary.valueType,
            "right operand type does not match binary result type",
            s"$location.right",
            binary.right.span
          )

      case comparison: Compare[?] =>
        validateExpression(
          comparison.left,
          parameters,
          s"$location.left",
          scope
        ) ++
          validateExpression(
            comparison.right,
            parameters,
            s"$location.right",
            scope
          ) ++
          requireSameType(
            comparison.left.valueType,
            comparison.operandType,
            "left operand type does not match comparison operand type",
            s"$location.left",
            comparison.left.span
          ) ++
          requireSameType(
            comparison.right.valueType,
            comparison.operandType,
            "right operand type does not match comparison operand type",
            s"$location.right",
            comparison.right.span
          )

      case intrinsic: Intrinsic[?] =>
        intrinsicTypes.get(intrinsic.name) match
          case None =>
            Vector(
              ValidationError(
                ValidationCode.UnknownIntrinsic,
                s"CUDA intrinsic '${intrinsic.name}' is not supported",
                location,
                intrinsic.span
              )
            )
          case Some(expectedType) =>
            requireSameType(
              intrinsic.valueType,
              expectedType,
              s"intrinsic '${intrinsic.name}' has type ${intrinsic.valueType.cudaName}, " +
                s"expected ${expectedType.cudaName}",
              location,
              intrinsic.span
            )

      case conversion: Convert[?, ?] =>
        validateExpression(
          conversion.value,
          parameters,
          s"$location.value",
          scope
        )

      case accumulation: ToAccumulator[?, ?] =>
        validateExpression(
          accumulation.value,
          parameters,
          s"$location.value",
          scope
        ) ++
          requireSameType(
            accumulation.value.valueType,
            accumulation.rule.inputType,
            "value type does not match accumulation input type",
            s"$location.value",
            accumulation.value.span
          )

      case index: ReductionIndex =>
        if scope.reductionIndexes.contains(index.name) then Vector.empty
        else
          Vector(
            ValidationError(
              ValidationCode.UnboundReductionIndex,
              s"reduction index '${index.name}' is used outside its reduction",
              location,
              index.span
            )
          )

      case index: LoopIndex =>
        if scope.loopIndexes.contains(index.name) then Vector.empty
        else
          Vector(
            ValidationError(
              ValidationCode.UnboundLoopIndex,
              s"loop index '${index.name}' is used outside its loop",
              location,
              index.span
            )
          )

      case reduction: ReduceSum[?, ?] =>
        val indexErrors =
          (if isIdentifier(reduction.index.name) then Vector.empty
           else
             Vector(
               ValidationError(
                 ValidationCode.InvalidReductionIndexName,
                 s"'${reduction.index.name}' is not a valid CUDA reduction index",
                 s"$location.index",
                 reduction.index.span
               )
             )) ++
            (if scope.reductionIndexes.contains(reduction.index.name) ||
                scope.loopIndexes.contains(reduction.index.name) ||
                scope.locals.contains(reduction.index.name)
             then
               Vector(
                 ValidationError(
                   ValidationCode.DuplicateReductionIndex,
                   s"reduction index '${reduction.index.name}' conflicts with an active binding",
                   s"$location.index",
                   reduction.index.span
                 )
               )
             else Vector.empty) ++
            (if parameters.contains(reduction.index.name) then
               Vector(
                 ValidationError(
                   ValidationCode.ReductionIndexConflictsWithParameter,
                   s"reduction index '${reduction.index.name}' conflicts with a parameter",
                   s"$location.index",
                   reduction.index.span
                 )
               )
             else Vector.empty)

        val rangeErrors =
          validateExpression(
            reduction.from,
            parameters,
            s"$location.from",
            scope
          ) ++
            validateExpression(
              reduction.until,
              parameters,
              s"$location.until",
              scope
            ) ++
            requireSameType(
              reduction.from.valueType,
              I32,
              "reduction lower bound must have CUDA int type",
              s"$location.from",
              reduction.from.span
            ) ++
            requireSameType(
              reduction.until.valueType,
              I32,
              "reduction upper bound must have CUDA int type",
              s"$location.until",
              reduction.until.span
            )

        val accumulatorErrors =
          validateExpression(
            reduction.initial,
            parameters,
            s"$location.initial",
            scope
          ) ++
            requireSameType(
              reduction.initial.valueType,
              reduction.rule.accumulatorType,
              "reduction initial value does not match its accumulator type",
              s"$location.initial",
              reduction.initial.span
            ) ++
            requireSameType(
              reduction.addition,
              reduction.rule.accumulatorType,
              "reduction addition capability does not match its accumulator type",
              location,
              reduction.span
            )

        val valueErrors =
          validateExpression(
            reduction.value,
            parameters,
            s"$location.value",
            scope.copy(
              reductionIndexes =
                scope.reductionIndexes + reduction.index.name
            )
          ) ++
            requireSameType(
              reduction.value.valueType,
              reduction.rule.inputType,
              "reduction value does not match its accumulation input type",
              s"$location.value",
              reduction.value.span
            )

        indexErrors ++ rangeErrors ++ accumulatorErrors ++ valueErrors

      case load: Load[?, ?, ?] =>
        validatePlace(
          load.from,
          parameters,
          s"$location.from",
          isWrite = false,
          scope = scope
        )

  private def validatePlace(
      place: Place[?, ?, ?],
      parameters: Map[String, KernelParam],
      location: String,
      isWrite: Boolean,
      scope: ValidationScope = ValidationScope()
  ): Vector[ValidationError] =
    place match
      case element: BufferElement[?, ?] =>
        val indexErrors =
          validateExpression(
            element.index,
            parameters,
            s"$location.index",
            scope
          ) ++
            requireSameType(
              element.index.valueType,
              I32,
              "buffer index must have CUDA int type",
              s"$location.index",
              element.index.span
            )

        val declarationErrors = parameters.get(element.bufferName) match
          case None =>
            Vector(
              ValidationError(
                ValidationCode.UnknownBuffer,
                s"buffer '${element.bufferName}' is not declared",
                location,
                element.span
              )
            )

          case Some(_: ScalarParam[?]) =>
            Vector(
              ValidationError(
                ValidationCode.ExpectedBuffer,
                s"parameter '${element.bufferName}' is scalar, not a buffer",
                location,
                element.span
              )
            )

          case Some(buffer: BufferParam[?, ?]) =>
            requireSameType(
              element.valueType,
              buffer.valueType,
              s"buffer element type ${element.valueType.cudaName} does not match " +
                s"parameter type ${buffer.valueType.cudaName}",
              location,
              element.span,
              ValidationCode.BufferTypeMismatch
            ) ++
              (if isWrite && buffer.access == BufferAccess.ReadOnly then
                 Vector(
                   ValidationError(
                     ValidationCode.WriteToReadOnlyBuffer,
                     s"buffer '${element.bufferName}' is read-only",
                     location,
                     element.span
                   )
                 )
               else Vector.empty)

        indexErrors ++ declarationErrors

      case local: LocalVariable[?] =>
        val nameErrors =
          if isIdentifier(local.name) then Vector.empty
          else
            Vector(
              ValidationError(
                ValidationCode.InvalidLocalName,
                s"'${local.name}' is not a valid CUDA local identifier",
                location,
                local.span
              )
            )

        val bindingErrors = scope.locals.get(local.name) match
          case None =>
            Vector(
              ValidationError(
                ValidationCode.UnboundLocal,
                s"local '${local.name}' is used before declaration or outside its scope",
                location,
                local.span
              )
            )
          case Some(declaredType) =>
            requireSameType(
              local.valueType,
              declaredType,
              s"local '${local.name}' has type ${local.valueType.cudaName}, " +
                s"but was declared as ${declaredType.cudaName}",
              location,
              local.span,
              ValidationCode.LocalTypeMismatch
            )

        nameErrors ++ bindingErrors

  private def validateLocalName(
      local: LocalVariable[?],
      parameters: Map[String, KernelParam],
      scope: ValidationScope,
      location: String
  ): Vector[ValidationError] =
    val identifierErrors =
      if isIdentifier(local.name) then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.InvalidLocalName,
            s"'${local.name}' is not a valid CUDA local identifier",
            s"$location.local",
            local.span
          )
        )

    val duplicateErrors =
      if scope.locals.contains(local.name) then
        Vector(
          ValidationError(
            ValidationCode.DuplicateLocalName,
            s"local '${local.name}' is already declared in an active scope",
            s"$location.local",
            local.span
          )
        )
      else Vector.empty

    val conflictErrors =
      if parameters.contains(local.name) ||
          scope.loopIndexes.contains(local.name) ||
          scope.reductionIndexes.contains(local.name)
      then
        Vector(
          ValidationError(
            ValidationCode.LocalNameConflictsWithBinding,
            s"local '${local.name}' conflicts with an active binding",
            s"$location.local",
            local.span
          )
        )
      else Vector.empty

    identifierErrors ++ duplicateErrors ++ conflictErrors

  private def validateLoopIndexName(
      index: LoopIndex,
      parameters: Map[String, KernelParam],
      scope: ValidationScope,
      location: String
  ): Vector[ValidationError] =
    val identifierErrors =
      if isIdentifier(index.name) then Vector.empty
      else
        Vector(
          ValidationError(
            ValidationCode.InvalidLoopIndexName,
            s"'${index.name}' is not a valid CUDA loop index",
            location,
            index.span
          )
        )

    val conflictErrors =
      if parameters.contains(index.name) ||
          scope.locals.contains(index.name) ||
          scope.loopIndexes.contains(index.name) ||
          scope.reductionIndexes.contains(index.name)
      then
        Vector(
          ValidationError(
            ValidationCode.LoopIndexConflictsWithBinding,
            s"loop index '${index.name}' conflicts with an active binding",
            location,
            index.span
          )
        )
      else Vector.empty

    identifierErrors ++ conflictErrors

  private def requireSameType(
      actual: CudaType[?],
      expected: CudaType[?],
      message: String,
      location: String,
      span: SourceSpan,
      code: ValidationCode = ValidationCode.ExpressionTypeMismatch
  ): Vector[ValidationError] =
    if actual == expected then Vector.empty
    else Vector(ValidationError(code, message, location, span))

  private def isIdentifier(value: String): Boolean =
    cudaIdentifier.matches(value)
