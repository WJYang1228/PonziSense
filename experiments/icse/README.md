# ICSE Experiments

Run from the repository root.

## Prepare the paper-aligned Ponzi-E split

```bash
python scripts/preprocess_ponzi_e_constructed.py \
  --input datafiles/ponzi_e_constructed/ponzi_e_real_contracts.csv \
  --output-dir datafiles/processed_ponzi_e
```

If you want existing `train.py` and `evaluate.py` to use this split directly:

```bash
python scripts/preprocess_ponzi_e_constructed.py \
  --input datafiles/ponzi_e_constructed/ponzi_e_real_contracts.csv \
  --output-dir datafiles/processed_ponzi_e \
  --also-write-default-processed
```

## Run all ICSE-oriented checks

```bash
bash scripts/run_icse_experiments.sh
```

Override paths when needed:

```bash
TEST_PATH=datafiles/processed_ponzi_e/test.csv \
TRAIN_PATH=datafiles/processed_ponzi_e/train.csv \
VAL_PATH=datafiles/processed_ponzi_e/val.csv \
FULL_PATH=datafiles/processed_ponzi_e/full_processed_with_groups.csv \
CHECKPOINT=outputs/checkpoints/best_model.pt \
OUTPUT_DIR=outputs \
DEVICE=cuda \
BATCH_SIZE=4 \
bash scripts/run_icse_experiments.sh
```

Useful runtime controls:

```bash
MAX_EVIDENCE_SAMPLES=100 \
MAX_ROLE_SAMPLES=80 \
GRAPH_WITH_ROLE_COVERAGE=1 \
STRESS_EVALS="random_test=datafiles/processed_ponzi_e/test.csv clone_heldout=datafiles/stress/clone_heldout.csv" \
bash scripts/run_icse_experiments.sh
```

## Outputs

- `outputs/icse/dataset_audit/`: split statistics, hash/template overlap, metadata coverage, rationale-field audit.
- `outputs/icse/dataset_stress_eval/`: PonziSense metrics on arbitrary externally prepared stress-test CSVs.
- `outputs/icse/refactor_robustness/`: identifier/comment/literal/layout robustness.
- `outputs/icse/mechanism_role_coverage/`: rationale coverage over Ponzi mechanism roles.
- `outputs/icse/syntax_preserving_faithfulness/`: graph perturbation without source deletion.
- `outputs/icse/evidence_chain_diagnostics/`: necessity and sufficiency diagnostics for predicted and annotated evidence chains.
- `outputs/icse/graph_component_ablation/`: inference-time CFG/DFG/propagation/source-only component diagnostics for PonziSense.
- `outputs/icse/efficiency/`: inference and explanation latency.
- `outputs/icse/summary/`: collected JSON/CSV index and `icse_tables_draft.md` for paper table transfer.

## What each script answers

| Script | Main review concern addressed | Requires checkpoint |
|---|---|---|
| `run_dataset_audit.py` | Dataset provenance, duplicate leakage, explanation-field coverage | No |
| `run_dataset_stress_eval.py` | Clone/template/temporal or custom held-out stress tests prepared as CSVs | Yes |
| `run_refactor_robustness.py` | Sensitivity to comments, layout, identifiers, literals | Yes |
| `run_mechanism_role_coverage.py` | Whether rationales cover Ponzi mechanism roles rather than isolated lines | Yes |
| `run_syntax_preserving_faithfulness.py` | Whether graph perturbation drops confidence without deleting source text | Yes |
| `run_evidence_chain_diagnostics.py` | Necessity and sufficiency of predicted/gold evidence chains against controls | Yes |
| `run_graph_component_ablation.py` | Whether CFG, DFG, propagation, and graph branch contribute distinct signals | Yes |
| `run_efficiency_benchmark.py` | Practical runtime and memory cost | Yes |
| `collect_icse_results.py` | One-folder result bundle for paper writing | No |

## Notes on interpretation

The graph ablation and evidence-chain scripts are inference-time diagnostics on the same trained PonziSense checkpoint. They are designed to answer whether the trained model uses the claimed graph and rationale signals. They are not a substitute for retrained ablation models. If the final paper claims retrained ablation, train separate checkpoints with the corresponding training flags and pass each checkpoint to the same scripts.
