# Ponzi-E Release Dataset

This directory contains a compact, source-free release view of the Ponzi-E benchmark.

- `ponzi_e_release.csv`: 8,233 rows with `contract_id`, `address`, `code_hash`, `label`, `explain`, `split`, `source`, and `explain_source`.
- `ponzi_e_release_report.json`: counts and address-coverage audit.

The Solidity `code` column is intentionally removed. Each row keeps a contract `address` plus `code_hash`, an irreversible identifier for audit and deduplication.

Label convention: `label=1` is Ponzi and `label=0` is non-Ponzi.
