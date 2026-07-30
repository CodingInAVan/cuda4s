package flight4s.core.launch

import scala.compiletime.testing.typeCheckErrors

import munit.FunSuite

class LaunchConfigSuite extends FunSuite:
  test("launch geometry helpers preserve all three CUDA dimensions"):
    assertEquals(Grid.x(7), Grid.xyz(7, 1, 1))
    assertEquals(Grid.xy(7, 5), Grid.xyz(7, 5, 1))
    assertEquals(Block.x(256), Block.xyz(256, 1, 1))
    assertEquals(Block.xy(32, 8), Block.xyz(32, 8, 1))
    assertEquals(Cluster.x(4), Cluster.xyz(4, 1, 1))
    assertEquals(Cluster.xy(2, 3), Cluster.xyz(2, 3, 1))

  test("every grid, block, and cluster dimension must be positive"):
    intercept[IllegalArgumentException](Grid.xyz(0, 1, 1))
    intercept[IllegalArgumentException](Grid.xyz(1, -1, 1))
    intercept[IllegalArgumentException](Grid.xyz(1, 1, 0))
    intercept[IllegalArgumentException](Block.xyz(-1, 1, 1))
    intercept[IllegalArgumentException](Block.xyz(1, 0, 1))
    intercept[IllegalArgumentException](Block.xyz(1, 1, -1))
    intercept[IllegalArgumentException](Cluster.xyz(0, 1, 1))
    intercept[IllegalArgumentException](Cluster.xyz(1, -1, 1))
    intercept[IllegalArgumentException](Cluster.xyz(1, 1, 0))

  test("private constructors prevent bypassing dimension validation"):
    val errors = typeCheckErrors(
      """
        import flight4s.core.launch.*
        val invalidGrid = Grid(1, 0, 1)
        val invalidBlock = Block(1, 1, 0)
        val invalidCluster = Cluster(0, 1, 1)
      """
    )

    assert(errors.nonEmpty)

  test("launch config carries dynamic shared memory requirements"):
    val config = LaunchConfig(
      grid = Grid.xyz(12, 3, 2),
      block = Block.xy(32, 8),
      dynamicSharedMemoryBytes = 4096,
      cluster = Some(Cluster.xyz(2, 3, 1))
    )

    assertEquals(config.grid, Grid.xyz(12, 3, 2))
    assertEquals(config.block, Block.xyz(32, 8, 1))
    assertEquals(config.dynamicSharedMemoryBytes, 4096)
    assertEquals(config.cluster, Some(Cluster.xyz(2, 3, 1)))

  test("dynamic shared memory size must not be negative"):
    intercept[IllegalArgumentException](
      LaunchConfig(Grid.x(1), Block.x(1), dynamicSharedMemoryBytes = -1)
    )

  test("grid dimensions must be divisible by cluster dimensions"):
    intercept[IllegalArgumentException](
      LaunchConfig(
        Grid.xyz(3, 4, 6),
        Block.x(1),
        cluster = Some(Cluster.x(2))
      )
    )
    intercept[IllegalArgumentException](
      LaunchConfig(
        Grid.xyz(4, 3, 6),
        Block.x(1),
        cluster = Some(Cluster.xy(2, 2))
      )
    )
    intercept[IllegalArgumentException](
      LaunchConfig(
        Grid.xyz(4, 4, 5),
        Block.x(1),
        cluster = Some(Cluster.xyz(2, 2, 2))
      )
    )
