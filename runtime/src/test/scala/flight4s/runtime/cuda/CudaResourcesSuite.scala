package flight4s.runtime.cuda

import java.nio.charset.StandardCharsets
import scala.collection.mutable.ArrayBuffer

import munit.FunSuite

import flight4s.core.codegen.*
import flight4s.core.compiler.*
import flight4s.core.dsl.CudaDsl.*
import flight4s.runtime.cuda.internal.*

class CudaResourcesSuite extends FunSuite:
  test("context owns modules and module closure invalidates functions"):
    val backend = RecordingBackend()
    val fixture = generatedFixture("ownedKernel")
    val context = openedContext(backend)
    val module = context.load(fixture.artifact).toOption.get
    val function = module.function(fixture.kernel).toOption.get

    assert(context.isOpen)
    assert(module.isOpen)
    assert(function.isValid)
    assertEquals(function.name, "ownedKernel")
    assert(function.signature eq fixture.kernel.signature)

    context.close()

    assert(!context.isOpen)
    assert(!module.isOpen)
    assert(!function.isValid)
    assertEquals(
      backend.events.toVector,
      Vector(
        "retain:0",
        "load:100",
        "resolve:100:200:ownedKernel",
        "unload:100:200",
        "release:0"
      )
    )

    context.close()
    module.close()
    assertEquals(backend.events.count(_ == "release:0"), 1)
    intercept[IllegalStateException](function.nativeHandle)

  test("module rejects kernels from a different generated artifact"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    val first = generatedFixture("firstKernel")
    val second = generatedFixture("secondKernel")
    val module = context.load(first.artifact).toOption.get

    intercept[IllegalArgumentException](
      module.function(second.kernel)
    )
    assert(!backend.events.exists(_.contains("secondKernel")))

    context.close()

  test("context closes multiple modules in reverse ownership order"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    context.load(generatedFixture("firstOwned").artifact).toOption.get
    context.load(generatedFixture("secondOwned").artifact).toOption.get

    context.close()

    assertEquals(
      backend.events.filter(_.startsWith("unload:")).toVector,
      Vector("unload:100:201", "unload:100:200")
    )

  test("module load and function lookup preserve structured failures"):
    val backend = RecordingBackend()
    val fixture = generatedFixture("failingKernel")
    val context = openedContext(backend)
    backend.loadStatus = failureStatus(
      "CUDA_ERROR_INVALID_PTX",
      errorLog = "invalid PTX at line 4"
    )

    val loadFailure = context.load(fixture.artifact).swap.toOption.get

    assertEquals(loadFailure.operation, "CUDA PTX module load")
    assertEquals(loadFailure.resultName, "CUDA_ERROR_INVALID_PTX")
    assertEquals(loadFailure.errorLog, "invalid PTX at line 4")

    backend.loadStatus = successStatus
    val module = context.load(fixture.artifact).toOption.get
    backend.resolveStatus = failureStatus("CUDA_ERROR_NOT_FOUND")

    val functionFailure =
      module.function(fixture.kernel).swap.toOption.get

    assert(functionFailure.operation.contains("failingKernel"))
    assertEquals(functionFailure.resultName, "CUDA_ERROR_NOT_FOUND")
    context.close()

  test("closed resources reject new operations"):
    val backend = RecordingBackend()
    val fixture = generatedFixture("closedKernel")
    val context = openedContext(backend)
    val module = context.load(fixture.artifact).toOption.get
    val function = module.function(fixture.kernel).toOption.get

    module.close()

    intercept[IllegalStateException](
      module.function(fixture.kernel)
    )
    intercept[IllegalStateException](function.nativeHandle)

    context.close()
    intercept[IllegalStateException](context.load(fixture.artifact))

  test("close failures retain ownership for a retry"):
    val backend = RecordingBackend()
    val fixture = generatedFixture("retryKernel")
    val context = openedContext(backend)
    val module = context.load(fixture.artifact).toOption.get
    backend.unloadStatus = failureStatus("CUDA_ERROR_INVALID_CONTEXT")

    val exception = intercept[CudaDriverException](context.close())

    assertEquals(
      exception.failure.operation,
      "CUDA module unload"
    )
    assert(context.isOpen)
    assert(module.isOpen)

    backend.unloadStatus = successStatus
    context.close()
    assert(!context.isOpen)

  test("context release failures leave the context open for a retry"):
    val backend = RecordingBackend()
    val context = openedContext(backend)
    backend.releaseStatus =
      failureStatus("CUDA_ERROR_INVALID_CONTEXT")

    val exception = intercept[CudaDriverException](context.close())

    assertEquals(
      exception.failure.operation,
      "CUDA primary context release"
    )
    assert(context.isOpen)

    backend.releaseStatus = successStatus
    context.close()
    assert(!context.isOpen)

  private def openedContext(
      backend: RecordingBackend
  ): CudaContext =
    CudaContext.open(0, backend).toOption.get

  private def generatedFixture(
      name: String
  ): GeneratedFixture =
    val definition = kernel(name, params()) { _ => () }
    val generatedKernel = CudaCodegen.generate(definition) match
      case Right(value) => value
      case Left(error) => fail(error.message)
    val generatedModule = GeneratedCudaModule(
      cudaSource = generatedKernel.cudaSource,
      sourceMap = generatedKernel.sourceMap,
      compilerOptions = generatedKernel.compilerOptions,
      kernels = Vector(generatedKernel)
    )
    val target = ComputeCapability(8, 0)
    val artifact = NvrtcArtifact(
      generated = generatedModule,
      ptx = IArray.unsafeFromArray(
        ".version 8.0".getBytes(StandardCharsets.UTF_8)
      ),
      compileLog = "",
      nvrtcVersion = NvrtcVersion(12, 0),
      target = target,
      compilerOptions = NvrtcCompileOptions.resolve(
        generatedModule.compilerOptions,
        target
      ),
      programName = s"$name.cu"
    )
    GeneratedFixture(generatedKernel, artifact)

  private val successStatus =
    NativeCudaDriverStatus(
      resultCode = 0,
      resultName = "CUDA_SUCCESS",
      resultDescription = "no error"
    )

  private def failureStatus(
      name: String,
      errorLog: String = ""
  ): NativeCudaDriverStatus =
    NativeCudaDriverStatus(
      resultCode = 1,
      resultName = name,
      resultDescription = "driver operation failed",
      errorLog = errorLog
    )

  private final case class GeneratedFixture(
      kernel: GeneratedKernel[EmptyTuple],
      artifact: NvrtcArtifact
  )

  private final class RecordingBackend extends CudaDriverBackend:
    val events: ArrayBuffer[String] = ArrayBuffer.empty
    var loadStatus: NativeCudaDriverStatus = successStatus
    var unloadStatus: NativeCudaDriverStatus = successStatus
    var resolveStatus: NativeCudaDriverStatus = successStatus
    var releaseStatus: NativeCudaDriverStatus = successStatus
    private var nextModuleHandle = 200L

    override def retainPrimaryContext(
        deviceOrdinal: Int
    ): NativeCudaContextRetainResult =
      events += s"retain:$deviceOrdinal"
      NativeCudaContextRetainResult(
        handle = 100L,
        deviceOrdinal = deviceOrdinal,
        computeCapabilityMajor = 8,
        computeCapabilityMinor = 9,
        status = successStatus
      )

    override def releasePrimaryContext(
        deviceOrdinal: Int
    ): NativeCudaDriverStatus =
      events += s"release:$deviceOrdinal"
      releaseStatus

    override def loadPtx(
        contextHandle: Long,
        ptx: IArray[Byte]
    ): NativeCudaResourceResult =
      events += s"load:$contextHandle"
      val handle = nextModuleHandle
      nextModuleHandle += 1
      NativeCudaResourceResult(
        handle = if loadStatus.succeeded then handle else 0L,
        status = loadStatus
      )

    override def unloadModule(
        contextHandle: Long,
        moduleHandle: Long
    ): NativeCudaDriverStatus =
      events += s"unload:$contextHandle:$moduleHandle"
      unloadStatus

    override def resolveFunction(
        contextHandle: Long,
        moduleHandle: Long,
        functionName: String
    ): NativeCudaResourceResult =
      events +=
        s"resolve:$contextHandle:$moduleHandle:$functionName"
      NativeCudaResourceResult(
        handle = if resolveStatus.succeeded then 300L else 0L,
        status = resolveStatus
      )
