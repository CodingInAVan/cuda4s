#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
NATIVE_DIR="$PROJECT_DIR/native"

export LD_LIBRARY_PATH="${NATIVE_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec sbt -Djava.library.path="$NATIVE_DIR" test
