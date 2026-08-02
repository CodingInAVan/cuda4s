package flight4s.runtime.cuda.internal

import munit.FunSuite

class InFlightLifetimeSuite extends FunSuite:
  test("close releases an idle resource exactly once"):
    var releaseCount = 0
    val state = InFlightResourceState[String] { () =>
      releaseCount += 1
      Right(())
    }

    assert(state.isOpen)
    assertEquals(state.requestClose(), Right(()))
    assert(!state.isOpen)
    assert(state.isReleased)
    assertEquals(releaseCount, 1)

    assertEquals(state.requestClose(), Right(()))
    assertEquals(releaseCount, 1)

  test("close waits for every in-flight lease"):
    var releaseCount = 0
    val state = InFlightResourceState[String] { () =>
      releaseCount += 1
      Right(())
    }
    val first = state.acquire()
    val second = state.acquire()

    assertEquals(state.requestClose(), Right(()))
    assert(!state.isOpen)
    assert(!state.isReleased)
    assertEquals(releaseCount, 0)
    intercept[IllegalStateException](state.acquire())

    assertEquals(first.release(), Right(()))
    assert(!state.isReleased)
    assertEquals(second.release(), Right(()))
    assert(state.isReleased)
    assertEquals(releaseCount, 1)

    assertEquals(second.release(), Right(()))
    assertEquals(releaseCount, 1)

  test("failed deferred release can be retried"):
    var shouldFail = true
    val state = InFlightResourceState[String] { () =>
      if shouldFail then Left("release failed") else Right(())
    }
    val lease = state.acquire()

    assertEquals(state.requestClose(), Right(()))
    assertEquals(lease.release(), Left("release failed"))
    assert(!state.isReleased)
    assert(state.isOpen)

    shouldFail = false
    assertEquals(state.requestClose(), Right(()))
    assert(state.isReleased)

  test("idle validation rejects active leases"):
    val state = InFlightResourceState[String](() => Right(()))
    val lease = state.acquire()

    intercept[IllegalStateException](state.requireIdle("pinned buffer"))
    assertEquals(lease.release(), Right(()))
    state.requireIdle("pinned buffer")

  test("batch completion releases every lease after one failure"):
    var firstFails = true
    var secondReleased = false
    val first = InFlightResourceState[String] { () =>
      if firstFails then Left("first failed") else Right(())
    }
    val second = InFlightResourceState[String] { () =>
      secondReleased = true
      Right(())
    }
    val firstLease = first.acquire()
    val secondLease = second.acquire()
    assertEquals(first.requestClose(), Right(()))
    assertEquals(second.requestClose(), Right(()))
    val batch = InFlightBatch(Vector(firstLease, secondLease))

    assertEquals(batch.complete(), Left("first failed"))
    assert(batch.isComplete)
    assert(secondReleased)

    firstFails = false
    assertEquals(first.requestClose(), Right(()))
    assert(first.isReleased)
    assertEquals(batch.complete(), Right(()))
