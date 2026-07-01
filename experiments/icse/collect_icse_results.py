from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def ensure_dir(path: str | Path) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def save_json(obj, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten(prefix: str, obj) -> dict:
    row = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            row.update(flatten(f"{prefix}.{key}" if prefix else str(key), value))
    elif isinstance(obj, list):
        row[prefix] = json.dumps(obj, ensure_ascii=False)
    else:
        row[prefix] = obj
    return row


def collect_files(icse_root: Path) -> tuple[list[dict], list[dict]]:
    json_rows = []
    csv_rows = []
    for path in sorted(icse_root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(icse_root)
        if path.suffix.lower() == ".json":
            try:
                payload = read_json(path)
                flat = flatten("", payload)
                json_rows.append({"file": str(rel), "experiment": rel.parts[0], **flat})
            except Exception as exc:
                json_rows.append({"file": str(rel), "experiment": rel.parts[0], "error": str(exc)})
        elif path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(path)
                csv_rows.append(
                    {
                        "file": str(rel),
                        "experiment": rel.parts[0],
                        "rows": int(len(df)),
                        "columns": ",".join(df.columns.astype(str).tolist()),
                    }
                )
            except Exception as exc:
                csv_rows.append({"file": str(rel), "experiment": rel.parts[0], "error": str(exc)})
    return json_rows, csv_rows


def markdown_from_known_outputs(icse_root: Path) -> str:
    parts = ["# ICSE Experiment Result Draft", ""]
    known_csvs = [
        ("Dataset Split Audit", icse_root / "dataset_audit" / "split_stats.csv"),
        ("Pairwise Split Overlap", icse_root / "dataset_audit" / "pairwise_overlap.csv"),
        ("Refactoring / Transformation Robustness", icse_root / "refactor_robustness" / "refactor_robustness_metrics.csv"),
        ("Dataset Stress Evaluation", icse_root / "dataset_stress_eval" / "dataset_stress_eval_metrics.csv"),
        ("Mechanism Role Coverage", icse_root / "mechanism_role_coverage" / "mechanism_role_coverage_samples.csv"),
        ("Evidence Necessity", icse_root / "evidence_chain_diagnostics" / "necessity_summary.csv"),
        ("Evidence Sufficiency", icse_root / "evidence_chain_diagnostics" / "sufficiency_summary.csv"),
        ("Graph Component Ablation", icse_root / "graph_component_ablation" / "graph_component_ablation_metrics.csv"),
        ("Efficiency", icse_root / "efficiency" / "forward_batch_latencies.csv"),
    ]
    for title, path in known_csvs:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        parts.append(f"## {title}")
        parts.append("")
        if len(df) > 20:
            df = df.head(20)
            parts.append("_Showing the first 20 rows._")
            parts.append("")
        try:
            parts.append(df.to_markdown(index=False))
        except Exception:
            parts.append("```csv")
            parts.append(df.to_csv(index=False).strip())
            parts.append("```")
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ICSE experiment outputs into a single summary directory.")
    parser.add_argument("--icse-root", default="./outputs/icse")
    parser.add_argument("--output-dir", default="./outputs/icse/summary")
    args = parser.parse_args()

    icse_root = Path(args.icse_root)
    if not icse_root.exists():
        raise FileNotFoundError(icse_root)
    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    json_rows, csv_rows = collect_files(icse_root)
    pd.DataFrame(json_rows).to_csv(out_root / "json_result_index.csv", index=False)
    pd.DataFrame(csv_rows).to_csv(out_root / "csv_result_index.csv", index=False)
    save_json({"json_files": json_rows, "csv_files": csv_rows}, out_root / "icse_results_bundle.json")

    md = markdown_from_known_outputs(icse_root)
    with open(out_root / "icse_tables_draft.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote summary files to {out_root}")


if __name__ == "__main__":
    main()
