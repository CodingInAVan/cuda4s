package flight4s.core.abi

import java.nio.{ByteBuffer, ByteOrder}

final case class NativeArgumentSlot(
    descriptor: KernelArgumentAbi,
    offsetBytes: Int
):
  require(offsetBytes >= 0, "native argument offset must not be negative")
  require(
    offsetBytes % descriptor.abiType.alignmentBytes == 0,
    s"${descriptor.name} offset $offsetBytes is not aligned to " +
      s"${descriptor.abiType.alignmentBytes} bytes"
  )

final class NativeArgumentStorage private (
    val slots: Vector[NativeArgumentSlot],
    private val storage: ByteBuffer
):
  require(storage.isDirect, "native argument storage must be direct")
  private val maxAlignment = slots
    .map(_.descriptor.abiType.alignmentBytes)
    .maxOption
    .getOrElse(1)
  require(
    storage.alignmentOffset(0, maxAlignment) == 0,
    s"native argument storage is not aligned to $maxAlignment bytes"
  )
  slots.zip(slots.drop(1)).foreach { (current, next) =>
    val currentEnd = Math.addExact(
      current.offsetBytes,
      current.descriptor.abiType.sizeBytes
    )
    require(
      currentEnd <= next.offsetBytes,
      s"${current.descriptor.name} overlaps ${next.descriptor.name}"
    )
  }
  private val requiredSize = slots.lastOption
    .map(slot =>
      Math.addExact(
        slot.offsetBytes,
        slot.descriptor.abiType.sizeBytes
      )
    )
    .getOrElse(0)
  require(
    storage.capacity() == requiredSize,
    s"native argument storage requires $requiredSize bytes, " +
      s"but received ${storage.capacity()}"
  )

  val sizeBytes: Int = storage.capacity()
  val offsets: Vector[Int] = slots.map(_.offsetBytes)
  val descriptorCodes: Vector[Byte] =
    slots.map(_.descriptor.abiType.nativeCode)

  private[flight4s] def bufferForLaunch: ByteBuffer =
    val view = storage.duplicate().order(ByteOrder.nativeOrder())
    view.position(0)
    view.limit(sizeBytes)
    view

object NativeArgumentStorage:
  private[flight4s] def materialize(
      packed: PackedKernelArguments
  ): NativeArgumentStorage =
    val slots = layout(packed.arguments)
    val sizeBytes = slots.lastOption
      .map(slot =>
        Math.addExact(
          slot.offsetBytes,
          slot.descriptor.abiType.sizeBytes
        )
      )
      .getOrElse(0)
    val maxAlignment = slots
      .map(_.descriptor.abiType.alignmentBytes)
      .maxOption
      .getOrElse(1)
    val storage = allocateAligned(sizeBytes, maxAlignment)

    packed.arguments.zip(slots).foreach { (argument, slot) =>
      storage.position(slot.offsetBytes)
      storage.put(argument.bytes.toArray)
    }
    storage.position(0)
    storage.limit(sizeBytes)

    new NativeArgumentStorage(slots, storage)

  private def layout(
      arguments: Vector[PackedKernelArgument]
  ): Vector[NativeArgumentSlot] =
    arguments
      .foldLeft((Vector.empty[NativeArgumentSlot], 0)) {
        case ((slots, nextOffset), argument) =>
          val offset = alignUp(
            nextOffset,
            argument.descriptor.abiType.alignmentBytes
          )
          val slot = NativeArgumentSlot(argument.descriptor, offset)
          val followingOffset = Math.addExact(
            offset,
            argument.descriptor.abiType.sizeBytes
          )
          (slots :+ slot, followingOffset)
      }
      ._1

  private def alignUp(offset: Int, alignment: Int): Int =
    require(
      alignment > 0 && Integer.bitCount(alignment) == 1,
      s"alignment must be a positive power of two: $alignment"
    )
    val remainder = offset & (alignment - 1)
    if remainder == 0 then offset
    else Math.addExact(offset, alignment - remainder)

  private def allocateAligned(
      sizeBytes: Int,
      alignmentBytes: Int
  ): ByteBuffer =
    if sizeBytes == 0 then
      ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder())
    else
      val allocationSize = Math.addExact(sizeBytes, alignmentBytes - 1)
      val allocation = ByteBuffer.allocateDirect(allocationSize)
      val aligned = allocation
        .alignedSlice(alignmentBytes)
        .order(ByteOrder.nativeOrder())
      aligned.limit(sizeBytes)
      aligned.slice().order(ByteOrder.nativeOrder())
