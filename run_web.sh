#!/usr/bin/env bash
# 在仓库根目录启动 Web 原型（system/）。用法: ./run_web.sh
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m system "$@"
