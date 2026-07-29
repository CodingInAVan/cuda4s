ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.8.1"
ThisBuild / organization := "com.cuda4s"

lazy val root = (project in file("."))
  .aggregate(core)
  .settings(
    name := "cuda4s-root",
    publish / skip := true
  )

lazy val core = (project in file("core"))
  .settings(
    name := "cuda4s-core",
    libraryDependencies += "org.scalameta" %% "munit" % "1.0.0" % Test
  )
