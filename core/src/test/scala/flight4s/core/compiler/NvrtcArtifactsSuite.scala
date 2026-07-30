package flight4s.core.compiler

import munit.FunSuite

import flight4s.core.codegen.CompilerOptions

class NvrtcArtifactsSuite extends FunSuite:
  test("compute capability produces the NVRTC virtual architecture"):
    val capability = ComputeCapability(8, 9)

    assertEquals(capability.major, 8)
    assertEquals(capability.minor, 9)
    assertEquals(capability.virtualArchitecture, "compute_89")
    assertEquals(
      capability.nvrtcOption,
      "--gpu-architecture=compute_89"
    )
    assertEquals(capability.toString, "8.9")

  test("compute capability dimensions are validated"):
    intercept[IllegalArgumentException](ComputeCapability(0, 0))
    intercept[IllegalArgumentException](ComputeCapability(8, -1))
    intercept[IllegalArgumentException](ComputeCapability(8, 10))

  test("resolved NVRTC options preserve order and append one target"):
    val generatedOptions = CompilerOptions(
      additionalNvrtcOptions = Vector(
        "--use_fast_math",
        "--device-debug"
      )
    )

    val resolved = NvrtcCompileOptions.resolve(
      generatedOptions,
      ComputeCapability(9, 0)
    )

    assertEquals(
      resolved.values,
      Vector(
        "--std=c++20",
        "--use_fast_math",
        "--device-debug",
        "--gpu-architecture=compute_90"
      )
    )

  test("NVRTC version rejects negative components"):
    assertEquals(NvrtcVersion(12, 8).toString, "12.8")
    intercept[IllegalArgumentException](NvrtcVersion(-1, 0))
    intercept[IllegalArgumentException](NvrtcVersion(12, -1))
