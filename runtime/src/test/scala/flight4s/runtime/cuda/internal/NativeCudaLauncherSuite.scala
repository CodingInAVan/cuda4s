package flight4s.runtime.cuda.internal

import java.nio.ByteBuffer

import munit.FunSuite

import flight4s.core.abi.CudaAbiType
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class NativeCudaLauncherSuite extends FunSuite:
  private val nativeLibraryConfigured =
    sys.props.contains("flight4s.cuda.native.path")

  test("JNI rejects unknown descriptor codes before CUDA launch"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val error = intercept[IllegalArgumentException](
      CudaNativeBindings.launchKernel(
        1L,
        1L,
        0L,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        false,
        0,
        0,
        0,
        ByteBuffer.allocateDirect(4),
        Array(0),
        Array(127.toByte)
      )
    )

    assert(error.getMessage.contains("unknown descriptor code"))

  test("JNI requires direct argument storage"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val error = intercept[IllegalArgumentException](
      CudaNativeBindings.launchKernel(
        1L,
        1L,
        0L,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        false,
        0,
        0,
        0,
        ByteBuffer.allocate(4),
        Array(0),
        Array(CudaAbiType.SignedInt32.nativeCode)
      )
    )

    assert(error.getMessage.contains("direct buffer"))

  test("JNI requires an exact argument buffer view"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val positioned = ByteBuffer.allocateDirect(4)
    positioned.position(1)
    val error = intercept[IllegalArgumentException](
      CudaNativeBindings.launchKernel(
        1L,
        1L,
        0L,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        false,
        0,
        0,
        0,
        positioned,
        Array(0),
        Array(CudaAbiType.SignedInt32.nativeCode)
      )
    )

    assert(error.getMessage.contains("position 0"))

  test("typed native request reaches JNI through the runtime wrapper"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val definition = kernel("empty", params()) { _ => () }
    val request = definition
      .bind(EmptyTuple)
      .nativeLaunchRequest(
        LaunchConfig(Grid.x(1), LaunchBlock.x(1))
      )

    val error = intercept[IllegalArgumentException](
      NativeCudaLauncher.launch(1L, 0L, 0L, request)
    )

    assert(error.getMessage.contains("function handle"))
