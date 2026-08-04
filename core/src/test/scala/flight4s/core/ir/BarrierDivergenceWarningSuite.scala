package flight4s.core.ir

import munit.FunSuite

import flight4s.core.codegen.CudaCodegen
import flight4s.core.dsl.CudaDsl.*

class BarrierDivergenceWarningSuite extends FunSuite:
  test("a barrier under thread-varying control flow warns without invalidating codegen"):
    val definition = kernel("unsafeBarrier", params()) { _ =>
      when(threadIdx.x < literal(32)) {
        barrier()
      }
    }

    val result = KernelValidator.validate(definition)

    assert(result.isValid)
    assertEquals(result.errors, Vector.empty)
    assertEquals(
      result.warnings.map(_.code),
      Vector(ValidationWarningCode.BarrierMayDiverge)
    )
    assertEquals(
      result.warnings.head.location,
      "body.statements[0].then.statements[0]"
    )
    assertNotEquals(result.warnings.head.span, SourceSpan.Unknown)
    assert(CudaCodegen.generate(definition).isRight)

  test("block-uniform branch conditions do not warn"):
    val definition = kernel("safeBarrier", params()) { _ =>
      when(blockIdx.x === literal(0)) {
        barrier()
      }
    }

    val result = KernelValidator.validate(definition)

    assert(result.isValid)
    assertEquals(result.warnings, Vector.empty)

  test("both paths of a divergent branch warn"):
    val definition = kernel("unsafeBarrierPaths", params()) { _ =>
      gpuIf(threadIdx.x < literal(32)) {
        barrier()
      } {
        barrier()
      }
    }

    val result = KernelValidator.validate(definition)

    assertEquals(
      result.warnings.map(_.location),
      Vector(
        "body.statements[0].then.statements[0]",
        "body.statements[0].else.statements[0]"
      )
    )

  test("nested divergent branches and divergent loops warn for each barrier"):
    val definition = kernel("nestedUnsafeBarriers", params()) { _ =>
      when(threadIdx.x < literal(32)) {
        when(blockIdx.x === literal(0)) {
          barrier()
        }
      }
      gpuFor("index", literal(0), threadIdx.x) { _ =>
        barrier()
      }
    }

    val result = KernelValidator.validate(definition)

    assertEquals(
      result.warnings.map(_.code),
      Vector(
        ValidationWarningCode.BarrierMayDiverge,
        ValidationWarningCode.BarrierMayDiverge
      )
    )
    assertEquals(
      result.warnings.map(_.location),
      Vector(
        "body.statements[0].then.statements[0].then.statements[0]",
        "body.statements[1].body.statements[0]"
      )
    )

  test("grid-uniform local conditions do not warn"):
    val definition = kernel("uniformLocalBarrier", params()) { _ =>
      val condition = local("condition", literal(true))
      when(condition.read) {
        barrier()
      }
    }

    val result = KernelValidator.validate(definition)

    assert(result.isValid)
    assertEquals(result.warnings, Vector.empty)

  test("varying local conditions warn"):
    val definition = kernel("varyingLocalBarrier", params()) { _ =>
      val lane = local("lane", threadIdx.x)
      when(lane.read < literal(32)) {
        barrier()
      }
    }

    val result = KernelValidator.validate(definition)

    assertEquals(
      result.warnings.map(_.code),
      Vector(ValidationWarningCode.BarrierMayDiverge)
    )

  test("block-uniform loop indexes can guard barriers"):
    val definition = kernel("uniformLoopBarrier", params()) { _ =>
      gpuFor("index", literal(0), blockIdx.x) { index =>
        when(index < literal(1)) {
          barrier()
        }
      }
    }

    val result = KernelValidator.validate(definition)

    assert(result.isValid)
    assertEquals(result.warnings, Vector.empty)

  test("module validation preserves barrier warnings with kernel locations"):
    val definition = kernel("moduleUnsafeBarrier", params()) { _ =>
      when(threadIdx.x < literal(32)) {
        barrier()
      }
    }

    val result = ModuleValidator.validate(CudaModuleIR(Vector.empty, Vector(definition.ir)))

    assert(result.isValid)
    assertEquals(
      result.warnings.map(_.code),
      Vector(ValidationWarningCode.BarrierMayDiverge)
    )
    assertEquals(
      result.warnings.head.location,
      "kernels[0].body.statements[0].then.statements[0]"
    )
