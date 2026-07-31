package flight4s.runtime.cuda

import java.nio.charset.StandardCharsets

import munit.FunSuite

import flight4s.core.codegen.*
import flight4s.core.compiler.NvrtcArtifact
import flight4s.core.dsl.CudaDsl.*
import flight4s.core.ir.Kernel
import flight4s.core.launch.{Block as LaunchBlock, Grid, LaunchConfig}

class CudaResourcesJniSuite extends FunSuite:
  private val nativeLibraryConfigured =
    sys.props.contains("flight4s.cuda.native.path")

  test("NVRTC artifact loads and resolves a typed CUDA function"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val context = openContext()
    var loadedModule = Option.empty[CudaModule]
    try
      val generated = generatedFixture("loadedKernel")
      val artifact = NvrtcCompiler.compile(
        generated.module,
        context.computeCapability,
        "loaded_kernel.cu"
      ) match
        case Right(value) => value
        case Left(failure) =>
          fail(failure.message + "\n" + failure.compileLog)

      val module = context.load(artifact) match
        case Right(value) => value
        case Left(failure) =>
          fail(failure.message + "\n" + failure.errorLog)
      loadedModule = Some(module)

      val function = module.function(generated.kernel) match
        case Right(value) => value
        case Left(failure) => fail(failure.message)

      assert(context.isOpen)
      assert(module.isOpen)
      assert(function.isValid)
      assertEquals(function.name, "loadedKernel")
      assert(function.signature eq generated.kernel.signature)
      assert(function.nativeHandle != 0L)

      function.launch(
        generated.definition.bind(EmptyTuple),
        LaunchConfig(Grid.x(1), LaunchBlock.x(1))
      ) match
        case Right(()) => ()
        case Left(failure) => fail(failure.message)

      module.close()
      assert(!module.isOpen)
      assert(!function.isValid)
      loadedModule = None
    finally
      loadedModule.foreach { module =>
        if module.isOpen then module.close()
      }
      if context.isOpen then context.close()

  test("Driver module failures retain JIT diagnostics"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val context = openContext()
    try
      val generated = generatedFixture("invalidPtxKernel")
      val validArtifact = NvrtcCompiler.compile(
        generated.module,
        context.computeCapability,
        "invalid_ptx_kernel.cu"
      ).toOption.get
      val invalidArtifact = validArtifact.copy(
        ptx = IArray.unsafeFromArray(
          "not valid PTX".getBytes(StandardCharsets.UTF_8)
        )
      )

      val failure =
        context.load(invalidArtifact).swap.toOption.get

      assertEquals(failure.operation, "CUDA PTX module load")
      assert(failure.resultCode != 0)
      assert(failure.resultName.startsWith("CUDA_ERROR_"))
      assert(failure.errorLog.nonEmpty)
    finally
      if context.isOpen then context.close()

  test("typed vectorAdd executes through an owned explicit stream"):
    assume(
      nativeLibraryConfigured,
      "set flight4s.cuda.native.path to run JNI tests"
    )

    val elementCount = 1024
    val leftValues =
      Array.tabulate(elementCount)(index => index.toFloat)
    val rightValues =
      Array.tabulate(elementCount)(index => (index * 2).toFloat)
    val expected =
      Array.tabulate(elementCount) { index =>
        leftValues(index) + rightValues(index)
      }

    val leftParam = input[Float]("left")
    val rightParam = input[Float]("right")
    val outputParam = output[Float]("output")
    val countParam = value[Int]("elementCount")
    val definition = kernel(
      "vectorAdd",
      params(leftParam, rightParam, outputParam, countParam)
    ) { bindings =>
      val left = bindings.head
      val right = bindings.tail.head
      val output = bindings.tail.tail.head
      val count = bindings.tail.tail.tail.head
      val index = local(
        "index",
        blockIdx.x * blockDim.x + threadIdx.x
      )
      when(index.read < count) {
        output(index.read) :=
          left(index.read).read + right(index.read).read
      }
    }
    val generatedKernel = CudaCodegen.generate(definition) match
      case Right(value) => value
      case Left(error) => fail(error.message)
    val generatedModule = GeneratedCudaModule(
      cudaSource = generatedKernel.cudaSource,
      sourceMap = generatedKernel.sourceMap,
      compilerOptions = generatedKernel.compilerOptions,
      kernels = Vector(generatedKernel)
    )

    val context = openContext()
    try
      val artifact = NvrtcCompiler.compile(
        generatedModule,
        context.computeCapability,
        "vector_add.cu"
      ) match
        case Right(value) => value
        case Left(failure) =>
          fail(failure.message + "\n" + failure.compileLog)
      val module = context.load(artifact) match
        case Right(value) => value
        case Left(failure) => fail(failure.message)
      val function = module.function(generatedKernel) match
        case Right(value) => value
        case Left(failure) => fail(failure.message)
      val stream = context.createStream().toOption.get
      val leftHost = context.allocatePinned[Float](elementCount).toOption.get
      val rightHost = context.allocatePinned[Float](elementCount).toOption.get
      val outputHost = context.allocatePinned[Float](elementCount).toOption.get
      val leftBuffer = context.allocate[Float](elementCount).toOption.get
      val rightBuffer = context.allocate[Float](elementCount).toOption.get
      val outputBuffer = context.allocate[Float](elementCount).toOption.get

      leftHost.copyFrom(leftValues)
      rightHost.copyFrom(rightValues)
      assertEquals(leftBuffer.copyFrom(leftHost), Right(()))
      assertEquals(rightBuffer.copyFrom(rightHost), Right(()))

      val invocation = definition.bind(
        (leftBuffer, rightBuffer, outputBuffer, elementCount)
      )
      function.launch(
        invocation,
        LaunchConfig(
          grid = Grid.x(elementCount / 256),
          block = LaunchBlock.x(256)
        ),
        stream
      ) match
        case Right(()) => ()
        case Left(failure) => fail(failure.message)

      stream.synchronize() match
        case Right(()) => ()
        case Left(failure) => fail(failure.message)

      outputBuffer.copyTo(outputHost) match
        case Right(()) => ()
        case Left(failure) => fail(failure.message)
      val actual = outputHost.toArray
      assertEquals(actual.toSeq, expected.toSeq)
    finally
      if context.isOpen then context.close()

  private def openContext(): CudaContext =
    CudaContext.open(0) match
      case Right(context) => context
      case Left(failure)
          if failure.resultName == "CUDA_ERROR_NO_DEVICE" =>
        assume(false, "CUDA device is not available")
        throw AssertionError("unreachable")
      case Left(failure) =>
        fail(failure.message)

  private def generatedFixture(
      name: String
  ): GeneratedFixture =
    val definition = kernel(name, params()) { _ => () }
    val generatedKernel = CudaCodegen.generate(definition) match
      case Right(value) => value
      case Left(error) => fail(error.message)
    GeneratedFixture(
      definition,
      generatedKernel,
      GeneratedCudaModule(
        cudaSource = generatedKernel.cudaSource,
        sourceMap = generatedKernel.sourceMap,
        compilerOptions = generatedKernel.compilerOptions,
        kernels = Vector(generatedKernel)
      )
    )

  private final case class GeneratedFixture(
      definition: Kernel[EmptyTuple],
      kernel: GeneratedKernel[EmptyTuple],
      module: GeneratedCudaModule
  )
