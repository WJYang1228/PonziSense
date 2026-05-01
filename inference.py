import os
import torch
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification

from configs.config import Config
from models.model import PonziModel
from data.feature_extractor import convert_code_to_features, build_attention_mask
from graph.statement_graph import build_statement_graph_tensors
from utils.statements import build_statement_labels


class PonziInferenceEngine:
    def __init__(self):
        self.cfg = Config()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cfg.DEVICE == "cuda" else "cpu"
        )

        self.tokenizer = RobertaTokenizer.from_pretrained(self.cfg.TOKENIZER_NAME)
        self.hf_config = RobertaConfig.from_pretrained(self.cfg.CONFIG_NAME)
        self.encoder = RobertaForSequenceClassification.from_pretrained(
            self.cfg.MODEL_NAME,
            config=self.hf_config
        )

        self.model = PonziModel(
            self.encoder, self.hf_config, num_clusters=self.cfg.NUM_CLUSTERS_K, cfg=self.cfg
        ).to(self.device)

        ckpt_path = os.path.join(self.cfg.CKPT_DIR, "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"未找到模型权重: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, code: str):
        """
        返回：
        {
            "pred_label": 0/1 (原始标签空间)
            "ponzi_prob": float,
            "non_ponzi_prob": float
        }
        """
        # 模型内部使用二分类标签：
        # feature.label = 1 if raw_label == POSITIVE_LABEL else 0
        # 所以 softmax[:,1] 对应 “positive class”
        feature = convert_code_to_features(
            code=code,
            label=0,   # 推理阶段占位，不影响真正预测
            explain="",
            tokenizer=self.tokenizer,
            cfg=self.cfg
        )
        attn_mask = build_attention_mask(feature, self.cfg, self.tokenizer)

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        position_idx = torch.tensor(feature.position_idx, dtype=torch.long).unsqueeze(0).to(self.device)
        attn_mask = torch.tensor(attn_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        stmts, _, _ = build_statement_labels(code, "")
        ga, gm, _ = build_statement_graph_tensors(code, self.cfg)
        graph_adj = torch.tensor(ga, dtype=torch.float32, device=self.device).unsqueeze(0)
        graph_mask = torch.tensor(gm, dtype=torch.float32, device=self.device).unsqueeze(0)

        probs, logits, outputs = self.model.forward_contract(
            input_ids,
            position_idx,
            attn_mask,
            labels=None,
            graph_adj=graph_adj,
            graph_mask=graph_mask,
            statements=[stmts],
            codes=[code],
            tokenizer=self.tokenizer,
        )
        probs = probs.squeeze(0).detach().cpu().tolist()

        positive_prob = float(probs[1])
        negative_prob = float(probs[0])

        # positive class 对应 cfg.POSITIVE_LABEL
        if positive_prob >= 0.5:
            raw_pred_label = self.cfg.POSITIVE_LABEL
        else:
            raw_pred_label = 1 if self.cfg.POSITIVE_LABEL == 0 else 0

        return {
            "pred_label": raw_pred_label,
            "ponzi_prob": positive_prob,
            "non_ponzi_prob": negative_prob,
        }