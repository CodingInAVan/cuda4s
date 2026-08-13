package flight4s.core.codegen

import munit.FunSuite

import flight4s.core.ir.ReductionPolicy

class ReductionLoweringStrategySuite extends FunSuite:
  test("every reduction policy selects an explicit serial strategy"):
    assertEquals(
      ReductionLoweringStrategy.select(ReductionPolicy.Strict),
      ReductionLoweringStrategy.StrictSerial
    )
    assertEquals(
      ReductionLoweringStrategy.select(ReductionPolicy.Deterministic),
      ReductionLoweringStrategy.DeterministicSerial
    )
    assertEquals(
      ReductionLoweringStrategy.select(ReductionPolicy.Fast),
      ReductionLoweringStrategy.FastSerialFallback
    )

  test("strategy markers describe the current lowering contract"):
    assertEquals(
      ReductionLoweringStrategy.StrictSerial.cudaMarker,
      "strict/serial-left-fold"
    )
    assertEquals(
      ReductionLoweringStrategy.DeterministicSerial.cudaMarker,
      "deterministic/serial-stable"
    )
    assertEquals(
      ReductionLoweringStrategy.FastSerialFallback.cudaMarker,
      "fast/serial-fallback"
    )
