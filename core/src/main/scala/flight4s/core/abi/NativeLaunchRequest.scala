package flight4s.core.abi

import java.nio.ByteBuffer

import flight4s.core.launch.LaunchConfig

private[flight4s] enum NativeLaunchValidationError:
  case StorageMustBeDirect
  case StorageViewMismatch(
      position: Int,
      limit: Int,
      capacity: Int
  )
  case MetadataCountMismatch(offsetCount: Int, descriptorCount: Int)
  case UnknownDescriptorCode(argumentIndex: Int, descriptorCode: Byte)
  case StorageBaseMisaligned(requiredAlignmentBytes: Int)
  case NegativeOffset(argumentIndex: Int, offsetBytes: Int)
  case ArgumentMisaligned(
      argumentIndex: Int,
      offsetBytes: Int,
      requiredAlignmentBytes: Int
  )
  case ArgumentsOverlap(
      previousArgumentIndex: Int,
      argumentIndex: Int
  )
  case ArgumentOutOfBounds(
      argumentIndex: Int,
      endOffsetBytes: Long,
      storageSizeBytes: Int
  )
  case StorageExtentMismatch(
      argumentEndOffsetBytes: Int,
      storageSizeBytes: Int
  )

private[flight4s] object NativeLaunchContract:
  def validateArgumentLayout(
      storage: ByteBuffer,
      offsets: Array[Int],
      descriptorCodes: Array[Byte]
  ): Either[NativeLaunchValidationError, Unit] =
    if !storage.isDirect then
      Left(NativeLaunchValidationError.StorageMustBeDirect)
    else if storage.position() != 0 || storage.limit() != storage.capacity() then
      Left(
        NativeLaunchValidationError.StorageViewMismatch(
          storage.position(),
          storage.limit(),
          storage.capacity()
        )
      )
    else if offsets.length != descriptorCodes.length then
      Left(
        NativeLaunchValidationError.MetadataCountMismatch(
          offsets.length,
          descriptorCodes.length
        )
      )
    else
      val abiTypes = new Array[CudaAbiType](descriptorCodes.length)
      var index = 0
      while index < descriptorCodes.length do
        CudaAbiType.fromNativeCode(descriptorCodes(index)) match
          case Some(abiType) => abiTypes(index) = abiType
          case None =>
            return Left(
              NativeLaunchValidationError.UnknownDescriptorCode(
                index,
                descriptorCodes(index)
              )
            )
        index += 1

      val maxAlignment = abiTypes
        .map(_.alignmentBytes)
        .maxOption
        .getOrElse(1)
      if storage.capacity() > 0 &&
          storage.alignmentOffset(0, maxAlignment) != 0
      then
        Left(
          NativeLaunchValidationError.StorageBaseMisaligned(maxAlignment)
        )
      else
        validateSlots(storage.capacity(), offsets, abiTypes)

  private def validateSlots(
      storageSizeBytes: Int,
      offsets: Array[Int],
      abiTypes: Array[CudaAbiType]
  ): Either[NativeLaunchValidationError, Unit] =
    var previousEnd = 0
    var index = 0
    while index < offsets.length do
      val offset = offsets(index)
      val abiType = abiTypes(index)
      if offset < 0 then
        return Left(
          NativeLaunchValidationError.NegativeOffset(index, offset)
        )
      if offset % abiType.alignmentBytes != 0 then
        return Left(
          NativeLaunchValidationError.ArgumentMisaligned(
            index,
            offset,
            abiType.alignmentBytes
          )
        )
      if offset < previousEnd then
        return Left(
          NativeLaunchValidationError.ArgumentsOverlap(index - 1, index)
        )

      val end = offset.toLong + abiType.sizeBytes.toLong
      if end > storageSizeBytes.toLong then
        return Left(
          NativeLaunchValidationError.ArgumentOutOfBounds(
            index,
            end,
            storageSizeBytes
          )
        )

      previousEnd = end.toInt
      index += 1

    if previousEnd != storageSizeBytes then
      Left(
        NativeLaunchValidationError.StorageExtentMismatch(
          previousEnd,
          storageSizeBytes
        )
      )
    else Right(())

final class NativeLaunchRequest private (
    val config: LaunchConfig,
    arguments: NativeArgumentStorage
):
  val gridX: Int = config.grid.x
  val gridY: Int = config.grid.y
  val gridZ: Int = config.grid.z
  val blockX: Int = config.block.x
  val blockY: Int = config.block.y
  val blockZ: Int = config.block.z
  val dynamicSharedMemoryBytes: Int = config.dynamicSharedMemoryBytes
  val usesCluster: Boolean = config.cluster.nonEmpty
  val clusterX: Int = config.cluster.fold(0)(_.x)
  val clusterY: Int = config.cluster.fold(0)(_.y)
  val clusterZ: Int = config.cluster.fold(0)(_.z)
  val argumentCount: Int = arguments.slots.size
  val argumentStorageSizeBytes: Int = arguments.sizeBytes

  private val launchOffsets = arguments.offsets.toArray
  private val launchDescriptorCodes = arguments.descriptorCodes.toArray

  private[flight4s] def argumentBuffer: ByteBuffer =
    arguments.bufferForLaunch

  private[flight4s] def argumentOffsets: Array[Int] =
    launchOffsets

  private[flight4s] def argumentDescriptorCodes: Array[Byte] =
    launchDescriptorCodes

object NativeLaunchRequest:
  private[flight4s] def materialize(
      config: LaunchConfig,
      arguments: NativeArgumentStorage
  ): NativeLaunchRequest =
    val request = new NativeLaunchRequest(config, arguments)
    NativeLaunchContract.validateArgumentLayout(
      request.argumentBuffer,
      request.argumentOffsets,
      request.argumentDescriptorCodes
    ) match
      case Right(()) => request
      case Left(error) =>
        throw new IllegalArgumentException(
          s"invalid native launch request: $error"
        )
