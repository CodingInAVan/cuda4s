package flight4s.core.ir

import scala.compiletime.testing.typeCheckErrors

import munit.FunSuite

import flight4s.core.dsl.{DslError, DslErrorCode}
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.types.*

class MemorySpaceSuite extends FunSuite:
  test("the DSL preserves module, kernel, and lexical memory ownership"):
    val coefficients = constantArray[Float]("coefficients", 64)
    val outputBuffer = output[Float]("output")
    val definition = kernel(
      "useMemorySpaces",
      params(outputBuffer)
    ) { bindings =>
      val output = bindings.head
      val tile =
        sharedArray2D[Float](
          "tile",
          rows = 32,
          columns = 32,
          rowStride = 33
        )
      val dynamicTile = dynamicSharedArray[Float]("dynamicTile")
      val scratch = localArray[Float]("scratch", 2)
      val row = threadIdx.y
      val column = threadIdx.x

      tile(row, column) := coefficients(column).read
      scratch(literal(0)) := tile(row, column).read
      dynamicTile(column) := scratch(literal(0)).read
      output(column) := dynamicTile(column).read
    }
    val moduleIr = module(
      constants = Vector(coefficients),
      kernels = Vector(definition)
    )

    assertEquals(moduleIr.constants, Vector(coefficients))
    assertEquals(moduleIr.kernels, Vector(definition.ir))
    assertEquals(
      definition.sharedMemory.map(_.name),
      Vector("tile", "dynamicTile")
    )
    val tileLayout = definition.sharedMemory.head.size
      .asInstanceOf[StaticSharedMemory[?]]
      .layout
    assertEquals(tileLayout.logicalDimensions, Vector(32, 32))
    assertEquals(tileLayout.physicalDimensions, Vector(32, 33))
    assertEquals(tileLayout.rowMajorStrides, Vector(33L, 1L))
    assertEquals(tileLayout.physicalElementCount, 32L * 33L)
    assertEquals(definition.sharedMemory.head.rankWitness.rank, 2)
    val localArrayDeclaration = definition.body.statements.head
      .asInstanceOf[LocalArrayDeclaration[Float]]
    assertEquals(localArrayDeclaration.array.name, "scratch")
    assertEquals(localArrayDeclaration.array.valueType, F32)
    assertEquals(localArrayDeclaration.array.elementCount, 2)
    assertNotEquals(localArrayDeclaration.array.span, SourceSpan.Unknown)
    assertNotEquals(localArrayDeclaration.span, SourceSpan.Unknown)
    assertEquals(ModuleValidator.validate(moduleIr), ValidationResult(Vector.empty))
    assertEquals(definition.params, Vector(outputBuffer))

  test("constant array elements are read-only at compile time"):
    val errors = typeCheckErrors(
      """
        import flight4s.core.dsl.CudaDsl.*

        val table = constantArray[Float]("table", 16)
        kernel("invalidConstantWrite") {
          table(literal(0)) := literal(1.0f)
        }
      """
    )

    assert(errors.nonEmpty)

  test("shared-array indexing rank is checked at compile time"):
    val errors = typeCheckErrors(
      """
        import flight4s.core.dsl.CudaDsl.*

        kernel("invalidSharedRank") {
          val tile = sharedArray2D[Float]("tile", 32, 32)
          val invalid = tile(literal(0)).read
        }
      """
    )

    assert(errors.nonEmpty)

    val rank3Errors = typeCheckErrors(
      """
        import flight4s.core.dsl.CudaDsl.*

        kernel("invalidSharedRank3") {
          val tile = sharedArray3D[Float]("tile", 8, 32, 32)
          val invalid = tile(literal(0), literal(0)).read
        }
      """
    )

    assert(rank3Errors.nonEmpty)

  test("three-dimensional shared arrays retain padded physical layout"):
    val definition = kernel("rank3Shared") {
      val tile =
        sharedArray3D[Float](
          "tile",
          depth = 8,
          rows = 32,
          columns = 32,
          rowStride = 33
        )

      tile(blockIdx.z, threadIdx.y, threadIdx.x) := literal(1.0f)
    }
    val allocation = definition.sharedMemory.head.size
      .asInstanceOf[StaticSharedMemory[?]]
    val layout = allocation.layout

    assertEquals(definition.sharedMemory.head.rankWitness.rank, 3)
    assertEquals(allocation.rank, 3)
    assertEquals(layout.logicalDimensions, Vector(8, 32, 32))
    assertEquals(layout.physicalDimensions, Vector(8, 32, 33))
    assertEquals(layout.rowMajorStrides, Vector(1056L, 33L, 1L))
    assertEquals(layout.physicalElementCount, 8L * 32L * 33L)
    assertEquals(
      KernelValidator.validate(definition),
      ValidationResult(Vector.empty)
    )

  test("shared-array phantom rank must match its allocation rank"):
    val errors = typeCheckErrors(
      """
        import flight4s.core.ir.*
        import flight4s.core.types.F32

        val invalid: SharedArray[Float, Rank1] =
          SharedArray(
            "tile",
            F32,
            StaticSharedMemory.twoDimensional(32, 32, 32)
          )
      """
    )

    assert(errors.nonEmpty)

  test("shared memory must be declared directly in the kernel body"):
    val error = intercept[DslError](
      kernel("nestedShared") {
        when(literal(true)) {
          sharedArray[Float]("nested", 32)
        }
      }
    )

    assert(error.getMessage.contains("directly in a kernel body"))
    assertEquals(
      error.code,
      DslErrorCode.SharedMemoryDeclarationOutsideKernelBody
    )

  test("shared-memory declarations validate names, sizes, and dynamic count"):
    val parameter = output[Float]("conflict")
    val definition = KernelIR(
      "invalidShared",
      params(parameter),
      Block(Vector.empty),
      Vector(
        SharedArray("bad-name", F32, StaticSharedMemory(0)),
        SharedArray("conflict", F32, DynamicSharedMemory),
        SharedArray("anotherDynamic", F32, DynamicSharedMemory),
        SharedArray("anotherDynamic", F32, StaticSharedMemory(4)),
        SharedArray(
          "badStride",
          F32,
          StaticSharedMemory.twoDimensional(32, 32, 31)
        )
      )
    )

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.InvalidSharedMemoryName))
    assert(codes.contains(ValidationCode.InvalidSharedMemoryElementCount))
    assert(codes.contains(ValidationCode.InvalidSharedMemoryLayout))
    assert(codes.contains(ValidationCode.SharedMemoryNameConflictsWithParameter))
    assert(codes.contains(ValidationCode.DuplicateSharedMemoryName))
    assert(codes.contains(ValidationCode.MultipleDynamicSharedDeclarations))

  test("local arrays cannot escape their lexical block"):
    val scratch = LocalArray("scratch", F32, 4)
    val definition = KernelIR(
      "localArrayScope",
      params(),
      Block(
        Vector(
          IfThen(
            literal(true),
            Block(Vector(LocalArrayDeclaration(scratch)))
          ),
          Store(
            LocalArrayElement("scratch", literal(0), F32),
            literal(1.0f)
          )
        )
      )
    )

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.UnknownLocalArray))

  test("memory references resolve against the correct owner and type"):
    val coefficients = ConstantArray("coefficients", F32, 16)
    val shared = SharedArray("tile", F32, StaticSharedMemory(16))
    val definition = KernelIR(
      "invalidReferences",
      params(),
      Block(
        Vector(
          Store(
            SharedElement("missingShared", Vector(literal(0)), F32),
            literal(1.0f)
          ),
          LocalDeclaration(
            LocalVariable("fromMissingConstant", F32),
            Load(ConstantElement("missingConstant", literal(0), F32))
          ),
          Store(
            SharedElement("tile", Vector(literal(0), literal(1)), I32),
            literal(1)
          ),
          LocalDeclaration(
            LocalVariable("wrongConstantType", I32),
            Load(ConstantElement("coefficients", literal(0), I32))
          )
        )
      ),
      Vector(shared)
    )
    val moduleIr = CudaModuleIR(Vector(coefficients), Vector(definition))

    val codes = ModuleValidator.validate(moduleIr).errors.map(_.code)

    assert(codes.contains(ValidationCode.UnknownSharedMemory))
    assert(codes.contains(ValidationCode.UnknownConstant))
    assert(codes.contains(ValidationCode.SharedMemoryTypeMismatch))
    assert(codes.contains(ValidationCode.ConstantTypeMismatch))
    assert(codes.contains(ValidationCode.SharedMemoryIndexRankMismatch))

  test("module validation rejects invalid and conflicting symbols"):
    val invalid = ConstantArray("bad-name", F32, 0)
    val duplicateA = ConstantArray("duplicate", F32, 1)
    val duplicateB = ConstantArray("duplicate", I32, 2)
    val conflictingConstant = ConstantArray("sameName", F32, 1)
    val firstKernel = KernelIR("sameName", params(), Block(Vector.empty))
    val secondKernel = KernelIR("sameName", params(), Block(Vector.empty))
    val moduleIr = CudaModuleIR(
      Vector(invalid, duplicateA, duplicateB, conflictingConstant),
      Vector(firstKernel, secondKernel)
    )

    val codes = ModuleValidator.validate(moduleIr).errors.map(_.code)

    assert(codes.contains(ValidationCode.InvalidConstantName))
    assert(codes.contains(ValidationCode.InvalidConstantElementCount))
    assert(codes.contains(ValidationCode.DuplicateConstantName))
    assert(codes.contains(ValidationCode.DuplicateKernelName))
    assert(codes.contains(ValidationCode.ModuleSymbolConflict))

  test("module constants cannot be shadowed by kernel-scope bindings"):
    val table = ConstantArray("table", F32, 16)
    def tableLoad(name: String): LocalDeclaration[Float] =
      LocalDeclaration(
        LocalVariable(name, F32),
        Load(ConstantElement("table", literal(0), F32))
      )

    val parameterKernel = KernelIR(
      "parameterShadow",
      params(value[Float]("table")),
      Block(Vector(tableLoad("parameterValue")))
    )
    val sharedKernel = KernelIR(
      "sharedShadow",
      params(),
      Block(Vector(tableLoad("sharedValue"))),
      Vector(SharedArray("table", F32, StaticSharedMemory(16)))
    )
    val localKernel = KernelIR(
      "localShadow",
      params(),
      Block(
        Vector(
          LocalDeclaration(LocalVariable("table", F32), literal(0.0f)),
          tableLoad("localValue")
        )
      )
    )
    val localArrayKernel = KernelIR(
      "localArrayShadow",
      params(),
      Block(
        Vector(
          LocalArrayDeclaration(LocalArray("table", F32, 4)),
          tableLoad("localArrayValue")
        )
      )
    )
    val loopIndex = LoopIndex("table")
    val loopKernel = KernelIR(
      "loopShadow",
      params(),
      Block(
        Vector(
          ForLoop(
            loopIndex,
            literal(0),
            literal(1),
            Block(Vector(tableLoad("loopValue")))
          )
        )
      )
    )
    val reductionIndex = ReductionIndex("table")
    val reductionKernel = KernelIR(
      "reductionShadow",
      params(),
      Block(
        Vector(
          LocalDeclaration(
            LocalVariable("sum", F32),
            ReduceSum(
              reductionIndex,
              literal(0),
              literal(1),
              literal(0.0f),
              literal(1.0f),
              summon[AccumulatorType[Float, Float]],
              summon[AdditiveType[Float]],
              ReductionPolicy.Strict
            )
          ),
          tableLoad("reductionValue")
        )
      )
    )
    val moduleIr = CudaModuleIR(
      Vector(table),
      Vector(
        parameterKernel,
        sharedKernel,
        localKernel,
        localArrayKernel,
        loopKernel,
        reductionKernel
      )
    )

    val errors = ModuleValidator
      .validate(moduleIr)
      .errors
      .filter(_.code == ValidationCode.ConstantNameShadowed)

    assertEquals(errors.size, 6)

  test("constant references require module context for validation"):
    val table = ConstantArray("table", F32, 16)
    val definition = KernelIR(
      "constantContext",
      params(),
      Block(
        Vector(
          LocalDeclaration(
            LocalVariable("loaded", F32),
            Load(ConstantElement("table", literal(0), F32))
          )
        )
      )
    )

    val standaloneCodes =
      KernelValidator.validate(definition).errors.map(_.code)
    val moduleResult =
      ModuleValidator.validate(CudaModuleIR(Vector(table), Vector(definition)))

    assertEquals(
      standaloneCodes,
      Vector(ValidationCode.ModuleContextRequired)
    )
    assertEquals(moduleResult, ValidationResult(Vector.empty))

  test("literal constant-array indexes must fit the declared element count"):
    val coefficients = ConstantArray("coefficients", F32, 2)
    val definition = KernelIR(
      "constantLiteralBounds",
      params(),
      Block(
        Vector(
          LocalDeclaration(
            LocalVariable("negative", F32),
            Load(ConstantElement("coefficients", literal(-1), F32))
          ),
          LocalDeclaration(
            LocalVariable("pastEnd", F32),
            Load(ConstantElement("coefficients", literal(2), F32))
          ),
          LocalDeclaration(
            LocalVariable("dynamic", F32),
            Load(ConstantElement("coefficients", threadIdx.x, F32))
          )
        )
      )
    )
    val module = CudaModuleIR(Vector(coefficients), Vector(definition))

    val errors = ModuleValidator.validate(module).errors

    assertEquals(
      errors.map(_.code),
      Vector(
        ValidationCode.ConstantIndexOutOfBounds,
        ValidationCode.ConstantIndexOutOfBounds
      )
    )
    assert(errors.head.message.contains("found -1"))
    assert(errors(1).message.contains("found 2"))

  test("local arrays require positive sizes and unique active names"):
    val first = LocalArray("scratch", F32, 4)
    val duplicate = LocalArray("scratch", F32, 0)
    val definition = KernelIR(
      "invalidLocalArrays",
      params(),
      Block(
        Vector(
          LocalArrayDeclaration(first),
          LocalArrayDeclaration(duplicate)
        )
      )
    )

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assert(codes.contains(ValidationCode.InvalidLocalArrayElementCount))
    assert(codes.contains(ValidationCode.DuplicateLocalName))

  test("an invalid local-array size does not hide the declaration from uses"):
    val scratch = LocalArray("scratch", F32, 0)
    val definition = KernelIR(
      "invalidLocalArraySize",
      params(),
      Block(
        Vector(
          LocalArrayDeclaration(scratch),
          Store(
            LocalArrayElement("scratch", literal(0), F32),
            literal(1.0f)
          )
        )
      )
    )

    val codes = KernelValidator.validate(definition).errors.map(_.code)

    assertEquals(
      codes,
      Vector(ValidationCode.InvalidLocalArrayElementCount)
    )

  test("literal local-array indexes must fit the declared element count"):
    val scratch = LocalArray("scratch", F32, 2)
    val definition = KernelIR(
      "localArrayLiteralBounds",
      params(),
      Block(
        Vector(
          LocalArrayDeclaration(scratch),
          Store(
            LocalArrayElement("scratch", literal(-1), F32),
            literal(1.0f)
          ),
          Store(
            LocalArrayElement("scratch", literal(2), F32),
            literal(1.0f)
          ),
          Store(
            LocalArrayElement("scratch", threadIdx.x, F32),
            literal(1.0f)
          )
        )
      )
    )

    val errors = KernelValidator.validate(definition).errors

    assertEquals(
      errors.map(_.code),
      Vector(
        ValidationCode.LocalArrayIndexOutOfBounds,
        ValidationCode.LocalArrayIndexOutOfBounds
      )
    )
    assert(errors.head.message.contains("found -1"))
    assert(errors(1).message.contains("found 2"))

  test("literal shared-memory indexes must fit logical rather than padded dimensions"):
    val tile =
      SharedArray(
        "tile",
        F32,
        StaticSharedMemory.twoDimensional(
          rows = 2,
          columns = 3,
          rowStride = 4
        )
      )
    val dynamic = SharedArray("dynamic", F32, DynamicSharedMemory)
    val definition = KernelIR(
      "sharedLiteralBounds",
      params(),
      Block(
        Vector(
          Store(
            SharedElement("tile", Vector(literal(2), literal(0)), F32),
            literal(1.0f)
          ),
          Store(
            SharedElement("tile", Vector(literal(0), literal(3)), F32),
            literal(1.0f)
          ),
          Store(
            SharedElement(
              "dynamic",
              Vector(literal(Int.MaxValue)),
              F32
            ),
            literal(1.0f)
          )
        )
      ),
      Vector(tile, dynamic)
    )

    val errors = KernelValidator.validate(definition).errors

    assertEquals(
      errors.map(_.code),
      Vector(
        ValidationCode.SharedMemoryIndexOutOfBounds,
        ValidationCode.SharedMemoryIndexOutOfBounds
      )
    )
    assert(errors.head.message.contains("index 0"))
    assert(errors.head.message.contains("found 2"))
    assert(errors(1).message.contains("index 1"))
    assert(errors(1).message.contains("found 3"))
