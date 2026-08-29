"""Train the Phase 3 baseline: a linear head on cached CLIP features.

    python -m src.train --features-dir data/features --out runs/baseline.pt

Plain BCE, no augmentation, on features from ``scripts/cache_features.py`` --
this is the deliberate control the Phase 1 harness is meant to show collapsing
under jpeg=30 / resize=0.25, not a serious model (HANDOFF.md, PLAN.md §5).

Reads ``{features_dir}/{split}/{embeddings.npy,labels.npy,meta.json}`` for
``--train-split`` (default ``train``) and, if present, ``--val-split``
(default ``val``) for model selection by AUROC. Writes a checkpoint consumed
by ``src.models.clip_baseline.CLIPLinearScorer``:

    {"state_dict": ..., "backbone": ..., "pretrained": ..., "embed_dim": ...,
     "config": {...}}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.models.semantic_head import LinearHead


def _load_features(features_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, dict] | None:
    d = features_dir / split
    emb_path, lab_path = d / "embeddings.npy", d / "labels.npy"
    if not (emb_path.exists() and lab_path.exists()):
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
    return np.load(emb_path), np.load(lab_path), meta


def train(
    features_dir: Path,
    out: Path,
    train_split: str = "train",
    val_split: str = "val",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)

    train_data = _load_features(features_dir, train_split)
    if train_data is None:
        raise SystemExit(
            f"no cached features at {features_dir / train_split}; run "
            f"scripts/cache_features.py --manifest data/manifests/{train_split}.csv "
            f"--out {features_dir / train_split} first"
        )
    X_train, y_train, meta = train_data
    val_data = _load_features(features_dir, val_split)

    embed_dim = X_train.shape[1]
    head = LinearHead(embed_dim).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).float().to(device)
    n = Xt.shape[0]

    if val_data is not None:
        X_val, y_val, _ = val_data
        Xv = torch.from_numpy(X_val).float().to(device)
    else:
        print("[warn] no val split cached; selecting the final-epoch checkpoint "
              "instead of the best-val one")
        Xv = y_val = None

    best_val_auroc = -1.0
    best_state = None
    history = []
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for epoch in range(epochs):
        head.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_x, batch_y = Xt[idx], yt[idx]
            opt.zero_grad()
            logits = head(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        head.eval()
        with torch.no_grad():
            train_probs = torch.sigmoid(head(Xt)).cpu().numpy()
        train_auroc = roc_auc_score(y_train, train_probs)

        if Xv is not None:
            with torch.no_grad():
                val_probs = torch.sigmoid(head(Xv)).cpu().numpy()
            val_auroc = roc_auc_score(y_val, val_probs)
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            val_auroc = None

        history.append({"epoch": epoch, "loss": epoch_loss, "train_auroc": train_auroc,
                         "val_auroc": val_auroc})
        print(f"epoch {epoch:3d}  loss {epoch_loss:.4f}  train_auroc {train_auroc:.4f}"
              + (f"  val_auroc {val_auroc:.4f}" if val_auroc is not None else ""))

    state_dict = best_state if best_state is not None else head.state_dict()

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": state_dict,
        "backbone": meta.get("backbone", "ViT-B-16"),
        "pretrained": meta.get("pretrained", "openai"),
        "embed_dim": embed_dim,
        "config": {
            "train_split": train_split, "val_split": val_split, "epochs": epochs,
            "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
            "seed": seed, "best_val_auroc": None if best_val_auroc < 0 else best_val_auroc,
        },
    }, out)

    result = {
        "out": str(out),
        "n_train": n,
        "n_val": None if Xv is None else int(Xv.shape[0]),
        "best_val_auroc": None if best_val_auroc < 0 else round(best_val_auroc, 4),
        "final_train_auroc": round(history[-1]["train_auroc"], 4),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--out", type=Path, default=Path("runs/baseline.pt"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    train(
        a.features_dir, a.out, train_split=a.train_split, val_split=a.val_split,
        epochs=a.epochs, lr=a.lr, weight_decay=a.weight_decay,
        batch_size=a.batch_size, device=a.device, seed=a.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
