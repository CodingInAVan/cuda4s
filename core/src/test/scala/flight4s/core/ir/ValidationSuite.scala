package flight4s.core.ir

import munit.FunSuite

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.types.*

class ValidationSuite extends FunSuite:
  test("a structurally valid kernel passes validation"):
    val source = input[Float16]("source")
    val result = output[Float]("result")
    val index = threadIdx.x
    val definition = kernel("promoteValues", params(source, result)) { bindings =>
      val boundSource = bindings.head
      val boundResult = bindings.tail.head
      when(index < literal(32)) {
        boundResult(index) := boundSource(index).read.toAccumulator
      }
    }

    assertEquals(KernelValidator.validate(definition), ValidationResult(Vector.empty))
    assertEquals(source.access, BufferAccess.ReadOnly)
    assertEquals(result.access, BufferAccess.ReadWrite)

  test("kernel and parameter names must be unique CUDA identifiers"):
    val first = input[Float]("source-value")
    val second = output[Float]("source-value")
    val definition =
      KernelIR("invalid kernel", params(first, second), Block(Vector.empty))

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.InvalidKernelName))
    assertEquals(codes.count(_ == ValidationCode.InvalidParameterName), 2)
    assert(codes.contains(ValidationCode.DuplicateParameterName))

  test("buffer references must resolve to buffer parameters"):
    val scalar = value[Float]("scale")
    val unknownTarget =
      BufferElement[Float, ReadWrite]("missing", literal(0), F32)
    val scalarTarget =
      BufferElement[Float, ReadWrite]("scale", literal(0), F32)
    val definition = KernelIR(
      "invalidReferences",
      params(scalar),
      Block(
        Vector(
          Store(unknownTarget, literal(1.0f)),
          Store(scalarTarget, literal(1.0f))
        )
      )
    )

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.UnknownBuffer))
    assert(codes.contains(ValidationCode.ExpectedBuffer))

  test("runtime validation catches forged buffer type and access mismatches"):
    val readOnly = input[Float]("source")
    val forgedWrite =
      BufferElement[Float, ReadWrite]("source", literal(0), F32)
    val wrongType =
      BufferElement[Int, ReadWrite]("source", literal(1), I32)
    val definition = KernelIR(
      "forgedAccess",
      params(readOnly),
      Block(
        Vector(
          Store(forgedWrite, literal(1.0f)),
          Store(wrongType, literal(1))
        )
      )
    )

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.WriteToReadOnlyBuffer))
    assert(codes.contains(ValidationCode.BufferTypeMismatch))

  test("validation result exposes an Either boundary"):
    val valid =
      KernelValidator.validate(KernelIR("empty", params(), Block(Vector.empty)))
    val invalid =
      KernelValidator.validate(KernelIR("not valid", params(), Block(Vector.empty)))

    assertEquals(valid.toEither, Right(()))
    assert(invalid.toEither.isLeft)

  test("only declared CUDA intrinsics are accepted"):
    val result = output[Float]("result")
    val unknownIndex = Intrinsic("warpIdx.x", I32)
    val definition = kernel("unknownIntrinsic", params(result)) { bindings =>
      bindings.head(unknownIndex) := literal(1.0f)
    }

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.UnknownIntrinsic))
