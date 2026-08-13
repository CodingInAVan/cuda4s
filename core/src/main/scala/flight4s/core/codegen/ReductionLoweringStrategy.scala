package flight4s.core.codegen

import flight4s.core.ir.ReductionPolicy

private[core] enum ReductionLoweringStrategy(val cudaMarker: String):
  case StrictSerial
      extends ReductionLoweringStrategy("strict/serial-left-fold")
  case DeterministicSerial
      extends ReductionLoweringStrategy("deterministic/serial-stable")
  case FastSerialFallback
      extends ReductionLoweringStrategy("fast/serial-fallback")

private[core] object ReductionLoweringStrategy:
  def select(policy: ReductionPolicy): ReductionLoweringStrategy =
    policy match
      case ReductionPolicy.Strict => StrictSerial
      case ReductionPolicy.Deterministic => DeterministicSerial
      case ReductionPolicy.Fast => FastSerialFallback
