"""
合约分析领域服务：封装推理与可解释性展示数据，供 Web 层调用。
"""
from __future__ import annotations

from system.bootstrap import ensure_algorithm_on_path
from system.demo_utils import highlight_code_lines

ensure_algorithm_on_path()

from predict import predict_one


class AnalysisService:
    """智能合约分析（含可解释性子模块输出）。"""

    def analyze(self, code: str, top_k: int = 5) -> dict:
        text = (code or "").strip()
        if not text:
            raise ValueError("请输入待分析的 Solidity 源码。")

        pred = predict_one(text, top_k=top_k)
        stmt_ids = [x["stmt_id"] for x in pred["explanations"]]
        highlighted_html = highlight_code_lines(text, stmt_ids)

        return {
            "risk": {
                "label": pred["prediction"],
                "ponzi_probability": pred["ponzi_prob"],
                "decision_threshold": pred["threshold"],
                "class_probabilities": pred.get("all_probs") or [],
            },
            "explainability": {
                "module_name": "语句级归因",
                "top_k": top_k,
                "statements": pred["explanations"],
            },
            "source_view": {
                "highlighted_html": highlighted_html,
                "line_count": len(text.splitlines()),
            },
        }
