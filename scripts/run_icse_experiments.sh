#!/usr/bin/env bash
set -euo pipefail

TEST_PATH="${TEST_PATH:-datafiles/processed_ponzi_e/test.csv}"
TRAIN_PATH="${TRAIN_PATH:-datafiles/processed_ponzi_e/train.csv}"
VAL_PATH="${VAL_PATH:-datafiles/processed_ponzi_e/val.csv}"
FULL_PATH="${FULL_PATH:-datafiles/processed_ponzi_e/full_processed_with_groups.csv}"
METADATA_PATH="${METADATA_PATH:-datafiles/ponzi_e_constructed/ponzi_e_real_contracts_metadata.csv}"
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/best_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_EVIDENCE_SAMPLES="${MAX_EVIDENCE_SAMPLES:-200}"
MAX_ROLE_SAMPLES="${MAX_ROLE_SAMPLES:-120}"
MAX_EFFICIENCY_CONTRACTS="${MAX_EFFICIENCY_CONTRACTS:-300}"
EXPLAIN_SAMPLES="${EXPLAIN_SAMPLES:-50}"
GRAPH_WITH_ROLE_COVERAGE="${GRAPH_WITH_ROLE_COVERAGE:-1}"
STRESS_EVALS="${STRESS_EVALS:-random_test=${TEST_PATH}}"

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

python experiments/icse/run_dataset_audit.py \
  --train-path "$TRAIN_PATH" \
  --val-path "$VAL_PATH" \
  --test-path "$TEST_PATH" \
  --full-path "$FULL_PATH" \
  --metadata-path "$METADATA_PATH" \
  --output-dir "$OUTPUT_DIR"

python experiments/icse/run_dataset_stress_eval.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --eval $STRESS_EVALS

python experiments/icse/run_refactor_robustness.py \
  --test-path "$TEST_PATH" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --save-transformed-csv

python experiments/icse/run_mechanism_role_coverage.py \
  --test-path "$TEST_PATH" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size 1 \
  --top-k 5

python experiments/icse/run_syntax_preserving_faithfulness.py \
  --test-path "$TEST_PATH" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size 1 \
  --max-samples "$MAX_EVIDENCE_SAMPLES" \
  --k-values 1 3 5 8 10 \
  --random-repeats 3 \
  --mode edge_dampen

python experiments/icse/run_evidence_chain_diagnostics.py \
  --test-path "$TEST_PATH" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size 1 \
  --max-samples "$MAX_EVIDENCE_SAMPLES" \
  --top-k 5 \
  --max-chain-nodes 8 \
  --random-repeats 3 \
  --mode edge_dampen

GRAPH_ROLE_ARGS=()
if [[ "$GRAPH_WITH_ROLE_COVERAGE" == "1" || "$GRAPH_WITH_ROLE_COVERAGE" == "true" ]]; then
  GRAPH_ROLE_ARGS=(--with-role-coverage --max-role-samples "$MAX_ROLE_SAMPLES")
fi

python experiments/icse/run_graph_component_ablation.py \
  --test-path "$TEST_PATH" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  "${GRAPH_ROLE_ARGS[@]}"

python experiments/icse/run_efficiency_benchmark.py \
  --test-path "$TEST_PATH" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --max-contracts "$MAX_EFFICIENCY_CONTRACTS" \
  --explain-samples "$EXPLAIN_SAMPLES" \
  --top-k 5

python experiments/icse/collect_icse_results.py \
  --icse-root "$OUTPUT_DIR/icse" \
  --output-dir "$OUTPUT_DIR/icse/summary"

echo "ICSE experiments completed. Results are under ${OUTPUT_DIR}/icse/"
