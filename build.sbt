ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.8.1"

lazy val root = (project in file("."))
  .aggregate(core, ml)
  .settings(
    name := "cuda4s-root"
  )

lazy val core = (project in file("core"))
  .settings(
    name := "cuda4s-core",
    libraryDependencies += "org.scalameta" %% "munit" % "1.0.0" % Test
  )

lazy val ml = (project in file("ml"))
  .dependsOn(core)
  .settings(
    name := "cuda4s-ml",
    libraryDependencies += "org.scalameta" %% "munit" % "1.0.0" % Test
  )
