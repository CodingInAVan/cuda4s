package com.cuda4s.core.ir

import com.cuda4s.core.types.{Bool, CudaType, I32}

enum ValidationCode:
  case InvalidKernelName
  case InvalidParameterName
  case DuplicateParameterName
  case InvalidLocalName
  case UnknownBuffer
  case ExpectedBuffer
  case BufferTypeMismatch
  case WriteToReadOnlyBuffer
  case ExpressionTypeMismatch
  case UnknownIntrinsic

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
      location: String
  ): Vector[ValidationError] =
    block.statements.zipWithIndex.flatMap { case (statement, index) =>
      validateStatement(statement, parameters, s"$location.statements[$index]")
    }

  private def validateStatement(
      statement: Stmt,
      parameters: Map[String, KernelParam],
      location: String
  ): Vector[ValidationError] =
    statement match
      case store: Store[?, ?] =>
        validatePlace(store.to, parameters, s"$location.to", isWrite = true) ++
          validateExpression(store.value, parameters, s"$location.value") ++
          requireSameType(
            store.to.valueType,
            store.value.valueType,
            s"store target type ${store.to.valueType.cudaName} does not match " +
              s"value type ${store.value.valueType.cudaName}",
            location,
            store.span
          )

      case branch: IfThen =>
        validateExpression(branch.condition, parameters, s"$location.condition") ++
          requireSameType(
            branch.condition.valueType,
            Bool,
            "GPU branch condition must have CUDA bool type",
            s"$location.condition",
            branch.condition.span
          ) ++
          validateBlock(branch.thenBlock, parameters, s"$location.then") ++
          branch.elseBlock.toVector.flatMap(validateBlock(_, parameters, s"$location.else"))

      case _: Barrier =>
        Vector.empty

  private def validateExpression(
      expression: Expr[?],
      parameters: Map[String, KernelParam],
      location: String
  ): Vector[ValidationError] =
    expression match
      case _: Literal[?] =>
        Vector.empty

      case binary: Binary[?] =>
        validateExpression(binary.left, parameters, s"$location.left") ++
          validateExpression(binary.right, parameters, s"$location.right") ++
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
        validateExpression(comparison.left, parameters, s"$location.left") ++
          validateExpression(comparison.right, parameters, s"$location.right") ++
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
        validateExpression(conversion.value, parameters, s"$location.value")

      case accumulation: ToAccumulator[?, ?] =>
        validateExpression(accumulation.value, parameters, s"$location.value") ++
          requireSameType(
            accumulation.value.valueType,
            accumulation.rule.inputType,
            "value type does not match accumulation input type",
            s"$location.value",
            accumulation.value.span
          )

      case load: Load[?, ?, ?] =>
        validatePlace(load.from, parameters, s"$location.from", isWrite = false)

  private def validatePlace(
      place: Place[?, ?, ?],
      parameters: Map[String, KernelParam],
      location: String,
      isWrite: Boolean
  ): Vector[ValidationError] =
    place match
      case element: BufferElement[?, ?] =>
        val indexErrors =
          validateExpression(element.index, parameters, s"$location.index") ++
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
