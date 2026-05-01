
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "datafiles" / "ponzi_e_constructed"
TARGET_POS = 744
TARGET_NEG = 7489

LOCAL_SOURCES = [
    ROOT / "datafiles" / "processed" / "full_processed_with_groups.csv",
    ROOT / "datafiles" / "PonziDataset_20221114_explain_augmented_negatives.csv",
    ROOT / "datafiles" / "PonziDataset_20221114_explain.csv",
    ROOT / "datafiles" / "PonziDataset_20221114.csv",
    ROOT / "datafiles" / "Ponzi_contracts.csv",
]

MECHANISM_KEYWORDS = [
    "msg.value", "payable", "transfer", ".send", ".call", "withdraw",
    "deposit", "payout", "reward", "bonus", "dividend", "referrer",
    "referral", "investor", "participant", "queue", "matrix", "level",
    "commission", "fee", "owner", "balance",
]

NEGATIVE_REJECT_TERMS = ["ponzi", "pyramid", "forsage", "doubler", "hyip"]


def normalize_code(code: Any) -> str:
    if pd.isna(code):
        return ""
    code = str(code).replace("\r\n", "\n").replace("\r", "\n").strip()
    return code


def canonical_hash(code: str) -> str:
    compact = re.sub(r"\s+", " ", code).strip().lower()
    return hashlib.sha256(compact.encode("utf-8", "ignore")).hexdigest()


def looks_like_solidity(code: str) -> bool:
    low = code.lower()
    return len(code) >= 80 and ("contract " in low or "library " in low or "interface " in low) and "function" in low


def mechanism_score(code: str) -> int:
    low = code.lower()
    score = sum(1 for kw in MECHANISM_KEYWORDS if kw in low)
    score += min(8, low.count("function"))
    if "while" in low or "for " in low or "for(" in low:
        score += 2
    if "[]" in code or ".push(" in low:
        score += 2
    return score


def quality_score(code: str, explain: str = "") -> float:
    n = len(code)
    lines = code.count("\n") + 1
    score = 0.0
    score += 40.0 if looks_like_solidity(code) else 0.0
    score += min(30.0, mechanism_score(code) * 2.0)
    score += 20.0 if explain.strip() else 0.0
    if 500 <= n <= 60000:
        score += 20.0
    elif 200 <= n <= 140000:
        score += 8.0
    if 20 <= lines <= 1600:
        score += 10.0
    return score


def extract_heuristic_rationale(code: str, max_items: int = 8) -> str:
    lines = code.split("\n")
    scored: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if not low.strip() or low.strip() in {"{", "}"}:
            continue
        s = sum(1 for kw in MECHANISM_KEYWORDS if kw in low)
        if ".transfer" in low or ".send" in low or ".call" in low:
            s += 3
        if "msg.value" in low or "payable" in low:
            s += 2
        if s > 0:
            scored.append((s, i, line.strip()))
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = [text for _, _, text in scored[:max_items]]
    return "\n---\n".join(picked)


def iter_local_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in LOCAL_SOURCES:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        code_col = "code" if "code" in df.columns else ("source_code" if "source_code" in df.columns else None)
        if not code_col or "label" not in df.columns:
            continue
        for idx, row in df.iterrows():
            code = normalize_code(row.get(code_col, ""))
            if not looks_like_solidity(code):
                continue
            raw_label = int(row.get("label"))
            explain = "" if "explain" not in df.columns or pd.isna(row.get("explain")) else str(row.get("explain", "")).strip()
            address = "" if "address" not in df.columns or pd.isna(row.get("address")) else str(row.get("address", "")).strip()
            out.append(
                {
                    "code": code,
                    "raw_label": raw_label,
                    "explain": explain,
                    "address": address,
                    "source": str(path.relative_to(ROOT)),
                    "source_row": int(idx),
                    "hash": canonical_hash(code),
                    "quality_score": quality_score(code, explain),
                    "mechanism_score": mechanism_score(code),
                }
            )
    return out


def dedupe_best(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        h = row["hash"]
        old = best.get(h)
        if old is None or (row["quality_score"], len(row.get("explain", ""))) > (old["quality_score"], len(old.get("explain", ""))):
            best[h] = row
    return list(best.values())


def select_positives(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    # The current local project uses label=1 as the 1,226-candidate Ponzi pool.
    candidates = [r for r in rows if int(r["raw_label"]) == 1]
    candidates = dedupe_best(candidates)
    candidates.sort(key=lambda r: (-r["quality_score"], -r["mechanism_score"], len(r["code"])))
    selected = candidates[:target]
    for r in selected:
        if not r.get("explain", "").strip():
            r["explain"] = extract_heuristic_rationale(r["code"])
            r["explain_source"] = "heuristic_generated_from_source"
        else:
            r["explain_source"] = "existing_explain_field"
        r["label"] = 1
    return selected


def select_local_negatives(rows: list[dict[str, Any]], positive_hashes: set[str]) -> list[dict[str, Any]]:
    candidates = [r for r in rows if int(r["raw_label"]) == 0 and r["hash"] not in positive_hashes]
    candidates = dedupe_best(candidates)
    # Keep broad real contracts, but slightly prefer code that does not look Ponzi-like.
    def neg_key(r: dict[str, Any]):
        low = r["code"].lower()
        reject_hits = sum(1 for t in NEGATIVE_REJECT_TERMS if t in low)
        return (reject_hits, r["mechanism_score"], -len(r["code"]))
    candidates.sort(key=neg_key)
    for r in candidates:
        r["label"] = 0
        r["explain"] = ""
        r["explain_source"] = "not_applicable_negative"
    return candidates


def sourcify_list_addresses(session: requests.Session, pages: int, limit: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    after = None
    for _ in range(pages):
        params = {"limit": str(limit), "sort": "desc"}
        if after:
            params["afterMatchId"] = str(after)
        resp = session.get("https://sourcify.dev/server/v2/contracts/1", params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json().get("results", [])
        if not batch:
            break
        rows.extend(batch)
        after = batch[-1].get("matchId")
        time.sleep(0.1)
    return rows


def sources_from_sourcify_contract(obj: dict[str, Any]) -> str:
    std = obj.get("stdJsonInput") or {}
    sources = std.get("sources") or {}
    parts = []
    for name, entry in sorted(sources.items()):
        content = entry.get("content") if isinstance(entry, dict) else None
        if content and isinstance(content, str):
            parts.append(f"// Source: {name}\n{content}")
    return "\n\n".join(parts).strip()


def fetch_sourcify_source(session: requests.Session, item: dict[str, Any]) -> dict[str, Any] | None:
    address = item.get("address")
    if not address:
        return None
    try:
        resp = session.get(
            f"https://sourcify.dev/server/v2/contract/1/{address}",
            params={"fields": "stdJsonInput,compilation"},
            timeout=40,
        )
        if resp.status_code != 200:
            return None
        obj = resp.json()
        comp = obj.get("compilation") or {}
        if str(comp.get("language", "")).lower() != "solidity":
            return None
        code = sources_from_sourcify_contract(obj)
        if not looks_like_solidity(code):
            return None
        low = code.lower()
        if any(term in low for term in NEGATIVE_REJECT_TERMS):
            return None
        return {
            "code": code,
            "raw_label": 0,
            "label": 0,
            "explain": "",
            "explain_source": "not_applicable_negative",
            "address": address,
            "source": "sourcify_v2_contracts_mainnet",
            "source_row": "",
            "hash": canonical_hash(code),
            "quality_score": quality_score(code, ""),
            "mechanism_score": mechanism_score(code),
            "verified_at": item.get("verifiedAt", ""),
            "match_id": item.get("matchId", ""),
        }
    except Exception:
        return None


def supplement_negatives_from_sourcify(existing_hashes: set[str], need: int, pages: int, workers: int) -> list[dict[str, Any]]:
    if need <= 0:
        return []
    session = requests.Session()
    session.headers.update({"User-Agent": "PonziSense dataset builder"})
    addresses = sourcify_list_addresses(session, pages=pages)
    out: list[dict[str, Any]] = []
    seen = set(existing_hashes)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_sourcify_source, session, item) for item in addresses]
        for fut in cf.as_completed(futs):
            row = fut.result()
            if not row:
                continue
            if row["hash"] in seen:
                continue
            seen.add(row["hash"])
            out.append(row)
            if len(out) >= need:
                break
    return out[:need]


def write_outputs(out_dir: Path, positives: list[dict[str, Any]], negatives: list[dict[str, Any]], target_pos: int, target_neg: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = positives + negatives
    df = pd.DataFrame([{"code": r["code"], "label": int(r["label"]), "explain": r.get("explain", "")} for r in final])
    meta_cols = [
        "label", "address", "source", "source_row", "hash", "quality_score", "mechanism_score",
        "explain_source", "verified_at", "match_id",
    ]
    meta = pd.DataFrame([{c: r.get(c, "") for c in meta_cols} for r in final])
    df.to_csv(out_dir / "ponzi_e_real_contracts.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(out_dir / "ponzi_e_real_contracts_metadata.csv", index=False, encoding="utf-8-sig")
    report = {
        "output_csv": str((out_dir / "ponzi_e_real_contracts.csv").relative_to(ROOT)),
        "metadata_csv": str((out_dir / "ponzi_e_real_contracts_metadata.csv").relative_to(ROOT)),
        "target": {"ponzi": target_pos, "non_ponzi": target_neg, "total": target_pos + target_neg},
        "actual": {
            "ponzi": int((df["label"] == 1).sum()),
            "non_ponzi": int((df["label"] == 0).sum()),
            "total": int(len(df)),
        },
        "positive_explain_sources": meta[meta["label"].eq(1)]["explain_source"].value_counts(dropna=False).to_dict(),
        "negative_sources": meta[meta["label"].eq(0)]["source"].value_counts(dropna=False).to_dict(),
        "label_convention": "label=1 Ponzi, label=0 non-Ponzi",
        "notes": [
            "Ponzi rows are selected from the local 1,226/1,281 label=1 candidate pool by code quality and Ponzi-mechanism signals.",
            "Missing positive explanations are filled by deterministic source-line heuristics and recorded in metadata; these are not expert annotations.",
            "Sourcify rows are verified Ethereum mainnet Solidity contracts used to supplement non-Ponzi examples.",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--target-pos", type=int, default=TARGET_POS)
    ap.add_argument("--target-neg", type=int, default=TARGET_NEG)
    ap.add_argument("--sourcify-pages", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-sourcify", action="store_true")
    args = ap.parse_args()

    local_rows = iter_local_rows()
    positives = select_positives(local_rows, args.target_pos)
    if len(positives) < args.target_pos:
        raise SystemExit(f"Only selected {len(positives)} positive rows, need {args.target_pos}")

    used_hashes = {r["hash"] for r in positives}
    negatives = select_local_negatives(local_rows, used_hashes)
    negatives = negatives[: args.target_neg]
    used_hashes |= {r["hash"] for r in negatives}

    missing = args.target_neg - len(negatives)
    if missing > 0 and not args.no_sourcify:
        extra = supplement_negatives_from_sourcify(used_hashes, missing, pages=args.sourcify_pages, workers=args.workers)
        negatives.extend(extra)

    write_outputs(args.out_dir, positives, negatives[: args.target_neg], args.target_pos, args.target_neg)


if __name__ == "__main__":
    main()
