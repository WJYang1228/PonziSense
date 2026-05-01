#!/usr/bin/env python3
"""Materialize a source-code CSV from the compact address-only Ponzi-E release file.

The release CSV does not contain Solidity source code. This helper fetches verified
source code for rows with an address and writes a `code,label,explain` CSV that can
be consumed by `preprocess_dataset.py`.

Sourcify is tried first because it does not require an API key. Etherscan is optional
and requires ETHERSCAN_API_KEY.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

SOURCIFY_CONTRACT = "https://sourcify.dev/server/v2/contract/{chain}/{address}?fields=stdJsonInput,compilation"
ETHERSCAN_SOURCE = "https://api.etherscan.io/api?module=contract&action=getsourcecode&address={address}&apikey={api_key}"


def http_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "PonziSense-release/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def source_from_sourcify(address: str, chain: int = 1) -> str:
    url = SOURCIFY_CONTRACT.format(chain=chain, address=urllib.parse.quote(address))
    data = http_json(url)
    std = data.get("stdJsonInput") or {}
    sources = std.get("sources") or {}
    parts = []
    for name, item in sorted(sources.items()):
        content = item.get("content") if isinstance(item, dict) else None
        if content:
            parts.append(f"// File: {name}\n{content}")
    return "\n\n".join(parts).strip()


def source_from_etherscan(address: str, api_key: str) -> str:
    url = ETHERSCAN_SOURCE.format(address=urllib.parse.quote(address), api_key=urllib.parse.quote(api_key))
    data = http_json(url)
    result = data.get("result") or []
    if not result or not isinstance(result, list):
        return ""
    source = result[0].get("SourceCode") or ""
    return source.strip()


def fetch_source(address: str, chain: int, api_key: str | None, sleep: float) -> tuple[str, str]:
    if not address:
        return "", "missing_address"
    try:
        source = source_from_sourcify(address, chain=chain)
        if source:
            return source, "sourcify"
    except Exception:
        pass
    if api_key:
        try:
            if sleep:
                time.sleep(sleep)
            source = source_from_etherscan(address, api_key=api_key)
            if source:
                return source, "etherscan"
        except Exception:
            pass
    return "", "not_found"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datafiles/ponzi_e_release.csv")
    parser.add_argument("--output", default="datafiles/PonziDataset_20221114_explain_augmented_negatives.csv")
    parser.add_argument("--report", default="datafiles/materialize_report.json")
    parser.add_argument("--chain", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests.")
    parser.add_argument("--sleep", type=float, default=0.21, help="Delay before Etherscan fallback requests.")
    args = parser.parse_args()

    api_key = os.getenv("ETHERSCAN_API_KEY")
    in_path = Path(args.input)
    out_path = Path(args.output)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    status_counts = {}
    with in_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if args.limit and i >= args.limit:
                break
            source, status = fetch_source((row.get("address") or "").strip(), args.chain, api_key, args.sleep)
            status_counts[status] = status_counts.get(status, 0) + 1
            if source:
                rows_out.append({
                    "code": source,
                    "label": row.get("label", ""),
                    "explain": row.get("explain", ""),
                })

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "label", "explain"])
        writer.writeheader()
        writer.writerows(rows_out)

    report = {
        "input": str(in_path),
        "output": str(out_path),
        "materialized_rows": len(rows_out),
        "status_counts": status_counts,
        "etherscan_fallback_enabled": bool(api_key),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
