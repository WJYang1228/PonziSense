import os
import math
import torch
from torch.utils.data import DataLoader

from torch.optim import AdamW
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

try:
    import optuna
except ImportError:
    optuna = None

from configs.config import Config
from data.augment import build_semantic_views
from data.dataset import load_datasets
from data.collate import collate_fn
from data.view_batch import batch_tensors_from_codes
from models.model import PonziModel
from utils.seed import set_seed
from utils.metrics import cls_metrics
from utils.io import ensure_dir, save_json
from utils.losses import compute_explainer_loss
from utils.explain_metrics import compute_explainability_macro
from utils.amp_helpers import autocast_from_config, grad_scaler_from_config
from utils.optional_objectives import supervised_contrastive_loss
from utils.infonce import infonce_symmetric


def _dataloader_kwargs(cfg, device, shuffle: bool):
    kw = {
        "batch_size": cfg.TRAIN_BATCH_SIZE if shuffle else cfg.EVAL_BATCH_SIZE,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.NUM_WORKERS > 0,
    }
    if cfg.NUM_WORKERS > 0:
        pf = getattr(cfg, "DATALOADER_PREFETCH_FACTOR", None)
        if pf is not None and int(pf) > 0:
            kw["prefetch_factor"] = int(pf)
    return kw


def _nb(device):
    """pin_memory=True 时非阻塞拷贝，与 DataLoader 重叠传输与计算。"""
    return device.type == "cuda"


def _forward_eval(
    model, tokenizer, batch, device, labels=None
):
    nb = _nb(device)
    kw = dict(
        graph_adj=batch["graph_adj"].to(device, non_blocking=nb),
        graph_mask=batch["graph_mask"].to(device, non_blocking=nb),
        statements=batch["statements"],
        codes=batch["codes"],
        tokenizer=tokenizer,
    )
    return model.forward_contract(
        batch["input_ids"].to(device, non_blocking=nb),
        batch["position_idx"].to(device, non_blocking=nb),
        batch["attn_mask"].to(device, non_blocking=nb),
        labels=labels,
        **kw,
    )


def evaluate(model, tokenizer, dataloader, device, cfg, cls_loss_fn, use_amp):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    total_exp_loss = 0.0
    autocast_cm = (
        autocast_from_config(cfg, device)
        if use_amp
        else torch.autocast(device_type="cuda", enabled=False)
    )
    nb = _nb(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=nb)
            position_idx = batch["position_idx"].to(device, non_blocking=nb)
            attn_mask = batch["attn_mask"].to(device, non_blocking=nb)
            labels = batch["labels"].to(device, non_blocking=nb)

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


def run_training(
    cfg: Config,
    *,
    run_test: bool = True,
    quiet: bool = False,
    optuna_trial=None,
):
    """
    执行完整训练。返回 dict：best_val_f1, best_epoch, history；若 run_test 则写 test_metrics。

    optuna_trial: 若传入，则每 epoch 向 Optuna report 验证 F1 并支持剪枝。
    """
    set_seed(cfg.RANDOM_SEED)

    ensure_dir(cfg.OUTPUT_DIR)
    ensure_dir(cfg.CKPT_DIR)
    ensure_dir(cfg.LOG_DIR)
    ensure_dir(cfg.PRED_DIR)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        if getattr(cfg, "CUDNN_BENCHMARK", False):
            torch.backends.cudnn.benchmark = True
    use_amp = cfg.USE_AMP and device.type == "cuda"
    scaler = grad_scaler_from_config(cfg, device)
    autocast_cm = autocast_from_config(cfg, device)

    tokenizer = RobertaTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
    config = RobertaConfig.from_pretrained(cfg.CONFIG_NAME)
    encoder = RobertaForSequenceClassification.from_pretrained(cfg.MODEL_NAME, config=config)

    model = PonziModel(encoder, config, num_clusters=cfg.NUM_CLUSTERS_K, cfg=cfg).to(device)
    if getattr(cfg, "USE_ENCODER_GRADIENT_CHECKPOINTING", False):
        model.encoder.roberta.gradient_checkpointing_enable()
    if getattr(cfg, "USE_TORCH_COMPILE", False) and hasattr(torch, "compile"):
        compile_mode = getattr(cfg, "TORCH_COMPILE_MODE", "reduce-overhead")
        model = torch.compile(model, mode=compile_mode)

    train_set, val_set, test_set = load_datasets(tokenizer, cfg)

    train_loader = DataLoader(train_set, **_dataloader_kwargs(cfg, device, shuffle=True))
    val_loader = DataLoader(val_set, **_dataloader_kwargs(cfg, device, shuffle=False))
    test_loader = DataLoader(test_set, **_dataloader_kwargs(cfg, device, shuffle=False))

    accum = max(1, cfg.GRADIENT_ACCUMULATION_STEPS)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_optimizer_steps = steps_per_epoch * cfg.EPOCHS
    warmup_steps = cfg.WARMUP_STEPS
    if cfg.WARMUP_RATIO > 0 and warmup_steps == 0:
        warmup_steps = int(total_optimizer_steps * cfg.WARMUP_RATIO)

    class_weights = torch.tensor(cfg.CLS_CLASS_WEIGHTS, dtype=torch.float, device=device)
    cls_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.LABEL_SMOOTHING)

    opt_kw = dict(lr=cfg.LR, eps=cfg.ADAM_EPSILON, weight_decay=cfg.WEIGHT_DECAY)
    if getattr(cfg, "ADAMW_FUSED", False) and device.type == "cuda":
        try:
            optimizer = AdamW(model.parameters(), fused=True, **opt_kw)
        except TypeError:
            optimizer = AdamW(model.parameters(), **opt_kw)
    else:
        optimizer = AdamW(model.parameters(), **opt_kw)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps
    )

    best_f1 = -1.0
    best_epoch = 0
    history = []
    patience_ctr = 0
    epochs_ran = 0

    for epoch in range(cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_exp_loss = 0.0
        y_true, y_pred = [], []

        optimizer.zero_grad(set_to_none=True)
        nb = _nb(device)
        loop = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", disable=quiet)
        for step, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device, non_blocking=nb)
            position_idx = batch["position_idx"].to(device, non_blocking=nb)
            attn_mask = batch["attn_mask"].to(device, non_blocking=nb)
            labels = batch["labels"].to(device, non_blocking=nb)

            with autocast_cm:
                probs, logits, outputs = _forward_eval(
                    model, tokenizer, batch, device, labels=None
                )
                cls_loss = cls_loss_fn(logits, labels)
                contract_cls = outputs[:, 0, :]

                need_aug = (cfg.USE_INFONCE_VIEWS and cfg.USE_CONTRASTIVE_LOSS) or (
                    cfg.USE_EXPLAIN_LOSS
                    and getattr(cfg, "EXPLAIN_LOSS_MODE", "bce") != "bce"
                )
                aug_outputs = []
                if need_aug:
                    view_count = max(2, int(getattr(cfg, "SEMANTIC_VIEW_COUNT", 3)))
                    semantic_views = [build_semantic_views(c, cfg, view_count) for c in batch["codes"]]
                    for view_idx in range(1, view_count):
                        aug_codes = [views[view_idx] for views in semantic_views]
                        ids_b, pos_b, attn_b = batch_tensors_from_codes(aug_codes, tokenizer, cfg)
                        ids_b = ids_b.to(device, non_blocking=nb)
                        pos_b = pos_b.to(device, non_blocking=nb)
                        attn_b = attn_b.to(device, non_blocking=nb)
                        aug_outputs.append(model._roberta_outputs(ids_b, pos_b, attn_b))

                if cfg.USE_CONTRASTIVE_LOSS:
                    if cfg.USE_INFONCE_VIEWS and aug_outputs:
                        zs = [contract_cls] + [out[:, 0, :] for out in aug_outputs]
                        losses = []
                        for i in range(len(zs)):
                            for j in range(i + 1, len(zs)):
                                losses.append(infonce_symmetric(zs[i], zs[j], cfg.TAU_CONTRASTIVE))
                        l_con = torch.stack(losses).sum() if losses else torch.tensor(0.0, device=device)
                    elif cfg.USE_SUPCON_FALLBACK:
                        l_con = supervised_contrastive_loss(
                            contract_cls, labels, cfg.TAU_CONTRASTIVE
                        )
                    else:
                        l_con = torch.tensor(0.0, device=device)
                else:
                    l_con = torch.tensor(0.0, device=device)

                l_clu = (
                    model.dec_clustering_loss(contract_cls, cfg.CLUSTER_LOSS_TEMP)
                    if cfg.USE_CLUSTER_LOSS
                    else torch.tensor(0.0, device=device)
                )
                L_rep = cls_loss + cfg.LAMBDA1 * l_con + cfg.LAMBDA2 * l_clu
                exp_loss = (
                    compute_explainer_loss(
                        model,
                        tokenizer,
                        batch,
                        device,
                        cfg,
                        outputs,
                        contract_outputs_b=aug_outputs[0] if aug_outputs else None,
                    )
                    if cfg.USE_EXPLAIN_LOSS
                    else torch.tensor(0.0, device=device)
                )
                loss = (L_rep + cfg.EXPLAIN_LOSS_WEIGHT * exp_loss) / accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_last_in_accum = (step + 1) % accum == 0
            is_end = (step + 1) == len(train_loader)
            if is_last_in_accum or is_end:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                preds = probs.argmax(dim=-1)
            running_loss += (
                (cls_loss + cfg.LAMBDA1 * l_con + cfg.LAMBDA2 * l_clu).detach()
                + cfg.EXPLAIN_LOSS_WEIGHT * exp_loss.detach()
            ).item()
            running_cls_loss += cls_loss.detach().item()
            running_exp_loss += exp_loss.detach().item()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

            loop.set_postfix(
                loss=(
                    cls_loss.item()
                    + cfg.LAMBDA1 * l_con.item()
                    + cfg.LAMBDA2 * l_clu.item()
                    + cfg.EXPLAIN_LOSS_WEIGHT * exp_loss.item()
                ),
                cls_loss=cls_loss.item(),
                exp_loss=exp_loss.item(),
            )

        epochs_ran = epoch + 1
        train_metrics = cls_metrics(y_true, y_pred)
        train_metrics["loss"] = running_loss / max(1, len(train_loader))
        train_metrics["cls_loss"] = running_cls_loss / max(1, len(train_loader))
        train_metrics["exp_loss"] = running_exp_loss / max(1, len(train_loader))
        train_metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        val_metrics = evaluate(model, tokenizer, val_loader, device, cfg, cls_loss_fn, use_amp)

        history.append({
            "epoch": epochs_ran,
            "train": train_metrics,
            "val": val_metrics,
        })

        if not quiet:
            print(f"[Epoch {epochs_ran}] train={train_metrics} | val={val_metrics}")

        val_f1 = val_metrics["f1"]
        if optuna_trial is not None and optuna is not None:
            optuna_trial.report(val_f1, step=epoch)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        improved = val_f1 > best_f1 + cfg.EARLY_STOPPING_MIN_DELTA
        if improved:
            best_f1 = val_f1
            best_epoch = epochs_ran
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(cfg.CKPT_DIR, "best_model.pt"))
        else:
            patience_ctr += 1

        if cfg.EARLY_STOPPING_PATIENCE > 0 and patience_ctr >= cfg.EARLY_STOPPING_PATIENCE:
            if not quiet:
                print(f"Early stopping at epoch {epochs_ran} (patience={cfg.EARLY_STOPPING_PATIENCE}).")
            break

    save_json(history, os.path.join(cfg.LOG_DIR, "history.json"))

    out = {
        "best_val_f1": best_f1,
        "best_epoch": best_epoch,
        "history": history,
        "epochs_ran": epochs_ran,
        "best_val_miou": None,
    }

    if run_test and os.path.isfile(os.path.join(cfg.CKPT_DIR, "best_model.pt")):
        model.load_state_dict(
            torch.load(os.path.join(cfg.CKPT_DIR, "best_model.pt"), map_location=device, weights_only=True),
            strict=False,
        )
        test_metrics = evaluate(model, tokenizer, test_loader, device, cfg, cls_loss_fn, use_amp)
        test_explain = compute_explainability_macro(
            model,
            tokenizer,
            test_loader,
            device,
            cfg,
            desc="TEST explain metrics (MSR/MSP/MIoU)",
        )
        test_metrics["explainability"] = test_explain
        save_json(test_metrics, os.path.join(cfg.LOG_DIR, "test_metrics.json"))
        if not quiet:
            print("Test:", test_metrics)
            print(
                "Explainability (test): MIoU={:.4f} MSP={:.4f} MSR={:.4f} "
                "(n={}, |GT|>0: {})".format(
                    test_explain["MIoU"],
                    test_explain["MSP"],
                    test_explain["MSR"],
                    test_explain["explain_n_samples"],
                    test_explain["explain_n_gt_positive"],
                )
            )
        out["test_metrics"] = test_metrics

    return out


def main():
    cfg = Config()
    run_training(cfg, run_test=True, quiet=False, optuna_trial=None)


if __name__ == "__main__":
    main()
