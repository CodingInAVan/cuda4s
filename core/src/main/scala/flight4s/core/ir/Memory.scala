package flight4s.core.ir

import flight4s.core.types.CudaType

sealed trait MemoryRank
sealed trait Rank1 extends MemoryRank
sealed trait Rank2 extends MemoryRank
sealed trait Rank3 extends MemoryRank

sealed trait MemoryRankWitness[Rank <: MemoryRank]:
  def rank: Int

object MemoryRankWitness:
  given rank1Witness: MemoryRankWitness[Rank1] with
    override val rank: Int = 1

  given rank2Witness: MemoryRankWitness[Rank2] with
    override val rank: Int = 2

  given rank3Witness: MemoryRankWitness[Rank3] with
    override val rank: Int = 3

sealed trait MemoryLayout[Rank <: MemoryRank]:
  def logicalDimensions: Vector[Int]
  def physicalDimensions: Vector[Int]
  def rank: Int

  final def physicalElementCount: Long =
    physicalDimensions.foldLeft(1L)(_ * _.toLong)

  final def rowMajorStrides: Vector[Long] =
    physicalDimensions.indices.map { index =>
      physicalDimensions
        .drop(index + 1)
        .foldLeft(1L)(_ * _.toLong)
    }.toVector

object MemoryLayout:
  private final case class OneDimensional(
      elementCount: Int
  ) extends MemoryLayout[Rank1]:
    override val logicalDimensions: Vector[Int] = Vector(elementCount)
    override val physicalDimensions: Vector[Int] = Vector(elementCount)
    override val rank: Int = 1

  private final case class TwoDimensional(
      rows: Int,
      columns: Int,
      rowStride: Int
  ) extends MemoryLayout[Rank2]:
    override val logicalDimensions: Vector[Int] = Vector(rows, columns)
    override val physicalDimensions: Vector[Int] = Vector(rows, rowStride)
    override val rank: Int = 2

  private final case class ThreeDimensional(
      depth: Int,
      rows: Int,
      columns: Int,
      rowStride: Int
  ) extends MemoryLayout[Rank3]:
    override val logicalDimensions: Vector[Int] =
      Vector(depth, rows, columns)
    override val physicalDimensions: Vector[Int] =
      Vector(depth, rows, rowStride)
    override val rank: Int = 3

  def oneDimensional(elementCount: Int): MemoryLayout[Rank1] =
    OneDimensional(elementCount)

  def twoDimensional(
      rows: Int,
      columns: Int,
      rowStride: Int
  ): MemoryLayout[Rank2] =
    TwoDimensional(rows, columns, rowStride)

  def threeDimensional(
      depth: Int,
      rows: Int,
      columns: Int,
      rowStride: Int
  ): MemoryLayout[Rank3] =
    ThreeDimensional(depth, rows, columns, rowStride)

sealed trait SharedMemorySize[Rank <: MemoryRank]:
  def rank: Int

final case class StaticSharedMemory[Rank <: MemoryRank] private[ir] (
    layout: MemoryLayout[Rank]
)(using val rankWitness: MemoryRankWitness[Rank]
) extends SharedMemorySize[Rank]:
  override def rank: Int = rankWitness.rank

object StaticSharedMemory:
  def apply(elementCount: Int): StaticSharedMemory[Rank1] =
    new StaticSharedMemory[Rank1](
      MemoryLayout.oneDimensional(elementCount)
    )

  def twoDimensional(
      rows: Int,
      columns: Int,
      rowStride: Int
  ): StaticSharedMemory[Rank2] =
    new StaticSharedMemory[Rank2](
      MemoryLayout.twoDimensional(rows, columns, rowStride)
    )

  def threeDimensional(
      depth: Int,
      rows: Int,
      columns: Int,
      rowStride: Int
  ): StaticSharedMemory[Rank3] =
    new StaticSharedMemory[Rank3](
      MemoryLayout.threeDimensional(
        depth,
        rows,
        columns,
        rowStride
      )
    )

case object DynamicSharedMemory extends SharedMemorySize[Rank1]:
  override val rank: Int = 1

final case class ConstantArray[T](
    name: String,
    valueType: CudaType[T],
    elementCount: Int,
    span: SourceSpan = SourceSpan.Unknown
)

final case class SharedArray[T, Rank <: MemoryRank](
    name: String,
    valueType: CudaType[T],
    size: SharedMemorySize[Rank],
    span: SourceSpan = SourceSpan.Unknown
)(using val rankWitness: MemoryRankWitness[Rank])

final case class LocalArray[T](
    name: String,
    valueType: CudaType[T],
    elementCount: Int,
    span: SourceSpan = SourceSpan.Unknown
)
