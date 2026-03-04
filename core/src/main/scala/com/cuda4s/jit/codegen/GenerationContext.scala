package com.cuda4s.jit.codegen

import scala.collection.mutable

private class GenerationContext(sb: StringBuilder):
  private var indentLevel = 0
  private val scopeStack = mutable.Stack[mutable.Set[String]]()

  scopeStack.push(mutable.Set.empty)

  def indent(): Unit = indentLevel += 1
  def outdent(): Unit = indentLevel -= 1

  def newline(): Unit = sb.append("\n")

  def line(str: String): Unit =
    sb.append("\t" * indentLevel).append(str).append("\n")

  def isDeclared(name: String): Boolean = scopeStack.exists(_.contains(name))

  def declare(name: String): Unit = scopeStack.top.add(name)

  def withScope(preDefined: Iterable[String] = Nil)(f: => Unit): Unit =
    val newScope = mutable.Set.from(preDefined)
    scopeStack.push(newScope)
    try f
    finally scopeStack.pop()

