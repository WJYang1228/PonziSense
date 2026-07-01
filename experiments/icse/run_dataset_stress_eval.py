from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.icse.icse_common import (
    ensure_dir,
    evaluate_classifier,
    get_device,
    load_cfg,
    load_model_tokenizer,
    make_dataset_from_csv,
    make_loader,
    save_json,
    set_reproducible_seed,
)


def parse_eval_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for spec in specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            path = spec
            name = Path(path).stem
        parsed.append((name.strip(), path.strip()))
    return parsed


def discover_csvs(eval_dir: str | None) -> list[tuple[str, str]]:
    if not eval_dir:
        return []
    root = Path(eval_dir)
    if not root.exists():
        return []
    found = []
    for path in sorted(root.glob("*.csv")):
        found.append((path.stem, str(path)))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PonziSense on arbitrary stress-test CSV files.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval",
        nargs="*",
        default=["random_test=datafiles/processed/test.csv"],
        help="Evaluation specs, each as name=path or plain path.",
    )
    parser.add_argument("--eval-dir", default=None, help="Optional directory containing additional CSV stress sets.")
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    cfg = load_cfg(args)
    device = get_device(cfg, args.device)
    model, tokenizer = load_model_tokenizer(cfg, device, checkpoint=args.checkpoint)

    evals = parse_eval_specs(args.eval) + discover_csvs(args.eval_dir)
    dedup = {}
    for name, path in evals:
        dedup[name] = path
    if not dedup:
        raise ValueError("No evaluation CSVs were provided.")

    out_root = Path(args.output_dir) / "icse" / "dataset_stress_eval"
    ensure_dir(out_root)

    rows = []
    for name, path in dedup.items():
        if not Path(path).exists():
            print(f"[skip] {name}: {path} does not exist")
            continue
        cfg.TEST_PATH = path
        dataset = make_dataset_from_csv(path, tokenizer, cfg)
        loader = make_loader(dataset, cfg, device, shuffle=False)
        metrics = evaluate_classifier(model, tokenizer, loader, cfg, device, threshold=cfg.PRED_THRESHOLD)
        metrics["eval_name"] = name
        metrics["path"] = path
        rows.append(metrics)
        print(f"[{name}] {metrics}")

    if not rows:
        raise ValueError("All provided evaluation CSVs were missing or invalid.")

    base = next((r for r in rows if r["eval_name"] in {"random_test", "test", "original"}), rows[0])
    for row in rows:
        row["delta_f1_vs_base"] = float(row["f1"] - base["f1"])
        row["delta_auprc_vs_base"] = float(row.get("auprc", 0.0) - base.get("auprc", 0.0))

    pd.DataFrame(rows).to_csv(out_root / "dataset_stress_eval_metrics.csv", index=False)
    save_json(
        {
            "checkpoint": args.checkpoint,
            "evaluations": rows,
            "note": "This script evaluates the same trained PonziSense checkpoint on externally prepared stress-test CSVs.",
        },
        out_root / "dataset_stress_eval_summary.json",
    )


if __name__ == "__main__":
    main()
