import os
import threading
import torch
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification

from configs.config import Config
from models.model import PonziModel
from data.feature_extractor import convert_code_to_features, build_attention_mask
from graph.edge_rationale import project_statement_scores_to_edges
from graph.statement_graph import build_statement_graph_tensors
from utils.rationale_extractor import node_perturbation_rationales
from utils.statements import build_statement_labels
from utils.paths import setup_demo_runtime, resolve_checkpoint_path


cfg = Config()
_load_lock = threading.Lock()
_model = None
_tokenizer = None
_device = None


def _ensure_predictor():
    global _model, _tokenizer, _device
    with _load_lock:
        if _model is not None:
            return
        setup_demo_runtime()
        ckpt = resolve_checkpoint_path()
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"???????: {ckpt}\n"
                "??????? best_model.pt ??:\n"
                "  <??????>/outputs/checkpoints/best_model.pt\n"
                "??????? PONZI_CKPT ???????"
            )

        _device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
        _tokenizer = RobertaTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
        hf_config = RobertaConfig.from_pretrained(cfg.CONFIG_NAME)
        encoder = RobertaForSequenceClassification.from_pretrained(cfg.MODEL_NAME, config=hf_config)
        _model = PonziModel(encoder, hf_config, num_clusters=cfg.NUM_CLUSTERS_K, cfg=cfg).to(_device)
        _model.load_state_dict(torch.load(ckpt, map_location=_device, weights_only=True), strict=False)
        _model.eval()


@torch.inference_mode()
def predict_one(code: str, top_k: int = 5):
    _ensure_predictor()
    tokenizer = _tokenizer
    model = _model
    device = _device

    feature = convert_code_to_features(
        code=code,
        label=cfg.POSITIVE_LABEL,
        explain="",
        tokenizer=tokenizer,
        cfg=cfg,
    )
    attn_mask = build_attention_mask(feature, cfg, tokenizer)

    input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0).to(device)
    position_idx = torch.tensor(feature.position_idx, dtype=torch.long).unsqueeze(0).to(device)
    attn_mask = torch.tensor(attn_mask, dtype=torch.bool).unsqueeze(0).to(device)

    statements, _, blocks = build_statement_labels(code, explain="")
    ga_np, gm_np, _ = build_statement_graph_tensors(code, cfg)
    graph_adj = torch.tensor(ga_np, dtype=torch.float32, device=device).unsqueeze(0)
    graph_mask = torch.tensor(gm_np, dtype=torch.float32, device=device).unsqueeze(0)

    probs, _, _ = model.forward_contract(
        input_ids,
        position_idx,
        attn_mask,
        labels=None,
        graph_adj=graph_adj,
        graph_mask=graph_mask,
        statements=[statements],
        codes=[code],
        tokenizer=tokenizer,
    )
    probs = probs.squeeze(0).cpu().tolist()

    # Internal class 1 is the raw POSITIVE_LABEL; by paper convention that is Ponzi.
    ponzi_prob = probs[1]
    is_ponzi = ponzi_prob >= cfg.PRED_THRESHOLD

    all_rationales = node_perturbation_rationales(
        model,
        tokenizer,
        input_ids,
        position_idx,
        attn_mask,
        graph_adj,
        graph_mask,
        statements,
        blocks,
        code,
        base_ponzi_prob=float(ponzi_prob),
        top_k=None,
    )
    explain_items = all_rationales[:top_k]

    edge_items: list = []
    if getattr(cfg, "EXPORT_EDGE_EXPLANATIONS", False) and all_rationales:
        score_by_stmt = {int(item["stmt_id"]): float(item["score"]) for item in all_rationales}
        n_eff = min(len(blocks), int(graph_mask.sum().item()), ga_np.shape[0])
        scores = [score_by_stmt.get(int(block.stmt_id), 0.0) for block in blocks[:n_eff]]
        edge_items = project_statement_scores_to_edges(
            ga_np,
            gm_np,
            blocks[:n_eff],
            scores,
            top_k=int(getattr(cfg, "EDGE_EXPLANATION_TOP_K", 25)),
        )

    out = {
        "prediction": "Ponzi" if is_ponzi else "Non-Ponzi",
        "ponzi_prob": float(ponzi_prob),
        "all_probs": probs,
        "threshold": cfg.PRED_THRESHOLD,
        "explanations": explain_items,
    }
    if edge_items:
        out["edge_explanations"] = edge_items
    return out


if __name__ == "__main__":
    demo_code = '''
    contract Demo {
        address public owner;
        struct Investor { address addr; uint amount; bool paid; }
        Investor[] public investors;
        uint public idx = 0;
        constructor() { owner = msg.sender; }
        function join() public payable {
            require(msg.value > 0);
            investors.push(Investor(msg.sender, msg.value, false));
            owner.transfer(msg.value / 10);
            while(address(this).balance >= investors[idx].amount * 2 && idx < investors.length){
                investors[idx].addr.transfer(investors[idx].amount * 2);
                investors[idx].paid = true;
                idx++;
            }
        }
    }
    '''
    out = predict_one(demo_code, top_k=5)
    print(out)
