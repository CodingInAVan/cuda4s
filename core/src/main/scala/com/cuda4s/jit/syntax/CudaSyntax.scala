package com.cuda4s.jit.syntax

object CudaSyntax:
  extension (sc: StringContext)
    def cuda(args: Any*): String =
      val parts = sc.parts.iterator
      val arguments = args.iterator
      val sb = new StringBuilder(parts.next())
      while arguments.hasNext do
        sb.append(arguments.next().toString)
        sb.append(parts.next())
      sb.toString().stripMargin
