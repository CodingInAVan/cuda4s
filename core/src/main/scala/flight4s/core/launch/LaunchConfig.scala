package flight4s.core.launch

final case class Grid private (x: Int, y: Int, z: Int):
  require(x > 0, s"grid x dimension must be positive: $x")
  require(y > 0, s"grid y dimension must be positive: $y")
  require(z > 0, s"grid z dimension must be positive: $z")

object Grid:
  def x(x: Int): Grid =
    xyz(x, 1, 1)

  def xy(x: Int, y: Int): Grid =
    xyz(x, y, 1)

  def xyz(x: Int, y: Int, z: Int): Grid =
    new Grid(x, y, z)

final case class Block private (x: Int, y: Int, z: Int):
  require(x > 0, s"block x dimension must be positive: $x")
  require(y > 0, s"block y dimension must be positive: $y")
  require(z > 0, s"block z dimension must be positive: $z")

object Block:
  def x(x: Int): Block =
    xyz(x, 1, 1)

  def xy(x: Int, y: Int): Block =
    xyz(x, y, 1)

  def xyz(x: Int, y: Int, z: Int): Block =
    new Block(x, y, z)

final case class Cluster private (x: Int, y: Int, z: Int):
  require(x > 0, s"cluster x dimension must be positive: $x")
  require(y > 0, s"cluster y dimension must be positive: $y")
  require(z > 0, s"cluster z dimension must be positive: $z")

object Cluster:
  def x(x: Int): Cluster =
    xyz(x, 1, 1)

  def xy(x: Int, y: Int): Cluster =
    xyz(x, y, 1)

  def xyz(x: Int, y: Int, z: Int): Cluster =
    new Cluster(x, y, z)

final case class LaunchConfig(
    grid: Grid,
    block: Block,
    dynamicSharedMemoryBytes: Int = 0,
    cluster: Option[Cluster] = None
):
  require(
    dynamicSharedMemoryBytes >= 0,
    s"dynamic shared memory must not be negative: $dynamicSharedMemoryBytes"
  )
  cluster.foreach { dimensions =>
    require(
      grid.x % dimensions.x == 0,
      s"grid x dimension ${grid.x} must be divisible by cluster x " +
        s"dimension ${dimensions.x}"
    )
    require(
      grid.y % dimensions.y == 0,
      s"grid y dimension ${grid.y} must be divisible by cluster y " +
        s"dimension ${dimensions.y}"
    )
    require(
      grid.z % dimensions.z == 0,
      s"grid z dimension ${grid.z} must be divisible by cluster z " +
        s"dimension ${dimensions.z}"
    )
  }
