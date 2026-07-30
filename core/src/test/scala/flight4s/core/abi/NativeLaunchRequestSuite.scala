package flight4s.core.abi

import java.nio.{ByteBuffer, ByteOrder}

import munit.FunSuite

import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.*
import flight4s.core.launch.{
  Block as LaunchBlock,
  Cluster,
  Grid,
  LaunchConfig
}

class NativeLaunchRequestSuite extends FunSuite:
  test("native launch request flattens geometry and argument metadata"):
    val signature = params(
      value[Int]("count"),
      value[Double]("scale"),
      in[Float]("input")
    )
    val definition = kernel("request", signature) { _ => () }
    val config = LaunchConfig(
      grid = Grid.xyz(18, 4, 2),
      block = LaunchBlock.xy(32, 8),
      dynamicSharedMemoryBytes = 2048,
      cluster = Some(Cluster.xyz(3, 2, 1))
    )
    val request = definition
      .bind(
        (
          37,
          1.25d,
          TestDeviceBuffer[Float](0x1122334455667788L)
        )
      )
      .nativeLaunchRequest(config)

    assertEquals(request.config, config)
    assertEquals((request.gridX, request.gridY, request.gridZ), (18, 4, 2))
    assertEquals((request.blockX, request.blockY, request.blockZ), (32, 8, 1))
    assertEquals(request.dynamicSharedMemoryBytes, 2048)
    assert(request.usesCluster)
    assertEquals(
      (request.clusterX, request.clusterY, request.clusterZ),
      (3, 2, 1)
    )
    assertEquals(request.argumentCount, 3)
    assertEquals(request.argumentStorageSizeBytes, 24)
    assertEquals(request.argumentOffsets.toSeq, Seq(0, 8, 16))
    assertEquals(
      request.argumentDescriptorCodes.toSeq,
      Seq(
        CudaAbiType.SignedInt32.nativeCode,
        CudaAbiType.Float64.nativeCode,
        CudaAbiType.DevicePointer64.nativeCode
      )
    )

    val buffer = request.argumentBuffer
    assert(buffer.isDirect)
    assertEquals(buffer.order(), ByteOrder.nativeOrder())
    assertEquals(buffer.getInt(0), 37)
    assertEquals(buffer.getDouble(8), 1.25d)
    assertEquals(buffer.getLong(16), 0x1122334455667788L)

  test("native launch request omits cluster attributes by default"):
    val definition = kernel("ordinary", params()) { _ => () }
    val request = definition
      .bind(EmptyTuple)
      .nativeLaunchRequest(
        LaunchConfig(Grid.x(1), LaunchBlock.x(1))
      )

    assert(!request.usesCluster)
    assertEquals(
      (request.clusterX, request.clusterY, request.clusterZ),
      (0, 0, 0)
    )

  test("native launch contract accepts empty argument storage"):
    val storage = ByteBuffer.allocateDirect(0)

    assertEquals(
      NativeLaunchContract.validateArgumentLayout(
        storage,
        Array.emptyIntArray,
        Array.emptyByteArray
      ),
      Right(())
    )

  test("native launch contract rejects invalid metadata counts and codes"):
    val storage = ByteBuffer.allocateDirect(4)

    assertEquals(
      NativeLaunchContract.validateArgumentLayout(
        storage,
        Array(0),
        Array.emptyByteArray
      ),
      Left(NativeLaunchValidationError.MetadataCountMismatch(1, 0))
    )
    assertEquals(
      NativeLaunchContract.validateArgumentLayout(
        storage,
        Array(0),
        Array(127.toByte)
      ),
      Left(NativeLaunchValidationError.UnknownDescriptorCode(0, 127.toByte))
    )

  test("native launch contract rejects invalid offsets and extents"):
    assertEquals(
      validate(4, Array(-1), CudaAbiType.SignedInt32),
      Left(NativeLaunchValidationError.NegativeOffset(0, -1))
    )
    assertEquals(
      validate(8, Array(1), CudaAbiType.Float64),
      Left(NativeLaunchValidationError.ArgumentMisaligned(0, 1, 8))
    )
    assertEquals(
      validate(
        8,
        Array(0, 4),
        CudaAbiType.Float64,
        CudaAbiType.SignedInt32
      ),
      Left(NativeLaunchValidationError.ArgumentsOverlap(0, 1))
    )
    assertEquals(
      validate(4, Array(0), CudaAbiType.Float64),
      Left(NativeLaunchValidationError.ArgumentOutOfBounds(0, 8L, 4))
    )
    assertEquals(
      validate(8, Array(0), CudaAbiType.SignedInt32),
      Left(NativeLaunchValidationError.StorageExtentMismatch(4, 8))
    )

  test("native launch contract requires an exact direct buffer view"):
    assertEquals(
      NativeLaunchContract.validateArgumentLayout(
        ByteBuffer.allocate(4),
        Array(0),
        Array(CudaAbiType.SignedInt32.nativeCode)
      ),
      Left(NativeLaunchValidationError.StorageMustBeDirect)
    )

    val positioned = ByteBuffer.allocateDirect(4)
    positioned.position(1)
    assertEquals(
      NativeLaunchContract.validateArgumentLayout(
        positioned,
        Array(0),
        Array(CudaAbiType.SignedInt32.nativeCode)
      ),
      Left(NativeLaunchValidationError.StorageViewMismatch(1, 4, 4))
    )

    val allocation = ByteBuffer.allocateDirect(9)
    allocation.position(1)
    val misaligned = allocation.slice()
    assertEquals(
      NativeLaunchContract.validateArgumentLayout(
        misaligned,
        Array(0),
        Array(CudaAbiType.Float64.nativeCode)
      ),
      Left(NativeLaunchValidationError.StorageBaseMisaligned(8))
    )

  private def validate(
      storageSizeBytes: Int,
      offsets: Array[Int],
      abiTypes: CudaAbiType*
  ): Either[NativeLaunchValidationError, Unit] =
    NativeLaunchContract.validateArgumentLayout(
      ByteBuffer.allocateDirect(storageSizeBytes),
      offsets,
      abiTypes.map(_.nativeCode).toArray
    )

  private final case class TestDeviceBuffer[T](rawAddress: Long)
      extends DeviceBuffer[T]:
    override private[flight4s] val deviceAddress: DeviceAddress =
      DeviceAddress.fromRaw(rawAddress)
