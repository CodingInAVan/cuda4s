package com.cuda4s.jit.utils

object BenchmarkUtils {
  /**
   * Times a function execution across warmup and benchmark iterations.
   *
   * @param label   The label to print for this benchmark.
   * @param warmup  Number of warmup iterations (not timed).
   * @param iters   Number of benchmark iterations (timed).
   * @param f       The code block to benchmark. It should return a Double (sink) to prevent JVM optimization.
   * @return The average time per iteration in milliseconds.
   */
  def timeMs(label: String, warmup: Int, iters: Int)(f: => Double): Double = {
    var w = 0
    var sink = 0.0
    while (w < warmup) {
      sink += f
      w += 1
    }

    var i = 0
    val t0 = System.nanoTime()
    while (i < iters) {
      sink += f
      i += 1
    }
    val t1 = System.nanoTime()

    val ms = (t1 - t0) / 1e6
    val avg = ms / iters
    println(f"$label%-40s total=$ms%.3f ms, avg=$avg%.3f ms  (sink=$sink%.3f)")
    avg
  }
}
