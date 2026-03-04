package com.cuda4s.jit.ir

enum Ty {
  case I32
  case U32
  case F32
  case F64
  case Ptr(inner: Ty)
}
