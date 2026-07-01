import os
import torch
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from configs.config import Config
from data.dataset import load_datasets
from data.collate import collate_fn
from models.model import PonziModel
from utils.metrics import cls_metrics
from utils.seed import set_seed
from utils.losses import compute_explainer_loss
from utils.explain_metrics import compute_explainability_macro


def _forward_eval(model, tokenizer, batch, device, labels=None):
    return model.forward_contract(
        batch["input_ids"].to(device),
        batch["position_idx"].to(device),
        batch["attn_mask"].to(device),
        labels=labels,
        graph_adj=batch["graph_adj"].to(device),
        graph_mask=batch["graph_mask"].to(device),
        statements=batch["statements"],
        codes=batch["codes"],
        tokenizer=tokenizer,
    )


def evaluate(model, tokenizer, dataloader, device, cfg, cls_loss_fn):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    total_exp_loss = 0.0
    use_amp = cfg.USE_AMP and device.type == "cuda"
    autocast_cm = torch.autocast(device_type="cuda", enabled=use_amp)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            labels = batch["labels"].to(device)

            with autocast_cm:
                probs, logits, outputs = _forward_eval(
                    model, tokenizer, batch, device, labels=None
                )
                cls_loss = cls_loss_fn(logits, labels)
                exp_loss = (
                    compute_explainer_loss(model, tokenizer, batch, device, cfg, outputs)
                    if cfg.USE_EXPLAIN_LOSS
                    else torch.tensor(0.0, device=device)
                )
                loss = cls_loss + cfg.EXPLAIN_LOSS_WEIGHT * exp_loss

            preds = probs.argmax(dim=-1)
            total_loss += loss.float().item()
            total_exp_loss += exp_loss.float().item()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    metrics = cls_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, len(dataloader))
    metrics["exp_loss"] = total_exp_loss / max(1, len(dataloader))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def main():
    cfg = Config()
    set_seed(cfg.RANDOM_SEED)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
    config = RobertaConfig.from_pretrained(cfg.CONFIG_NAME)
    encoder = RobertaForSequenceClassification.from_pretrained(cfg.MODEL_NAME, config=config)

    model = PonziModel(encoder, config, num_clusters=cfg.NUM_CLUSTERS_K, cfg=cfg).to(device)

    ckpt_path = os.path.join(cfg.CKPT_DIR, "best_model.pt")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)

    _, val_set, test_set = load_datasets(tokenizer, cfg)

    loader_kw = {
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.NUM_WORKERS > 0,
    }
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kw,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kw,
    )

    class_weights = torch.tensor(cfg.CLS_CLASS_WEIGHTS, dtype=torch.float, device=device)
    cls_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.LABEL_SMOOTHING)

    val_metrics = evaluate(model, tokenizer, val_loader, device, cfg, cls_loss_fn)
    test_metrics = evaluate(model, tokenizer, test_loader, device, cfg, cls_loss_fn)

    val_explain = compute_explainability_macro(
        model, tokenizer, val_loader, device, cfg, desc="VAL explain metrics"
    )
    test_explain = compute_explainability_macro(
        model, tokenizer, test_loader, device, cfg, desc="TEST explain metrics"
    )
    val_metrics["explainability"] = val_explain
    test_metrics["explainability"] = test_explain

    print("VAL:", val_metrics)
    print("TEST:", test_metrics)


if __name__ == "__main__":
    main()
