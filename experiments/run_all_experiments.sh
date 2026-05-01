#!/usr/bin/env bash
# 在仓库根目录执行：
#   bash experiments/run_all_experiments.sh
# 并行（默认 1；多卡或显存极大时可加大，单卡并行易 OOM）：
#   PARALLEL_JOBS=2 bash experiments/run_all_experiments.sh
#
# 日志：outputs/logs/experiments/<时间戳>_<名称>.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
LOG="${REPO_ROOT}/outputs/logs/experiments"
mkdir -p "$LOG"
TS="$(date +%Y%m%d_%H%M%S)"
MAX_JOBS="${PARALLEL_JOBS:-1}"

# 启用作业控制，便于 wait -n 限制并发
set -m

run_one() {
  local name=$1
  shift
  echo "========================================"
  echo ">>> ${name}"
  echo "========================================"
  if "$@" 2>&1 | tee "${LOG}/${TS}_${name}.log"; then
    echo "[OK] ${name}"
  else
    echo "[FAIL] ${name}" >&2
    return 1
  fi
}

wait_slot() {
  while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n 2>/dev/null || sleep 0.2
  done
}

FAILED=0
launch() {
  local name=$1
  shift
  wait_slot
  (
    run_one "$name" "$@" || echo "FAILED:${name}" >> "${LOG}/${TS}_failed.txt"
  ) &
}

rm -f "${LOG}/${TS}_failed.txt"

if [[ "$MAX_JOBS" -gt 1 ]]; then
  echo "PARALLEL_JOBS=${MAX_JOBS} — 单 GPU 并行可能导致 OOM，请自行观察 nvidia-smi。"
fi

launch rq1_main python experiments/rq1/run_main_metrics.py --split test --threshold 0.5
launch rq2_overlap python experiments/rq2/run_overlap_metrics.py --split test
launch rq2_fd_arl python experiments/rq2/run_fd_arl.py --split test --max-samples 200
launch rq2_faith_curve python experiments/rq2/run_faithfulness_curve.py --split test --max-samples 100 --k-max 10
launch rq2_faith_ctrl python experiments/rq2/run_faithfulness_curve_controls.py --split test --max-samples 80 --k-list 1,2,3,5,8,10 --random-repeats 8
launch rq4_threshold python experiments/rq4/run_threshold_sweep.py --split test --thresholds 0.3,0.5,0.7
launch rq5_umap python experiments/rq5/run_umap_embeddings.py --split test --max-samples 800
launch case_latency python experiments/case_study/benchmark_latency.py --n 50

wait

if [[ -f "${LOG}/${TS}_failed.txt" ]]; then
  echo "部分任务失败，见 ${LOG}/${TS}_failed.txt" >&2
  FAILED=1
fi

if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi

echo "RQ3 消融图（依赖已生成的 RQ1/RQ2 JSON）..."
python experiments/rq3/run_ablation_plot.py \
  --manifest experiments/rq3/ablation_manifest.example.json \
  --out-tag ablation

echo "案例研究占位 PDF..."
python experiments/case_study/render_case_study_figure.py

echo "同步论文 figure/ 路径（PonziSense/figure/*.pdf）..."
python experiments/sync_paper_figures.py --split test --rq3-tag ablation

echo "全部完成。日志目录: ${LOG}"
echo "导出 LaTeX（outputs/paper_latex/）..."
python experiments/export_paper_latex.py --split test
echo "LaTeX 宏与表格: ${REPO_ROOT}/outputs/paper_latex/"
echo "论文插图 PDF: ${REPO_ROOT}/PonziSense/figure/"
