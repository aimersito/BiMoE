#!/usr/bin/env python3
"""Entrenamiento BiMoE-10K. Corre en Kaggle (T4) o local con PyTorch CPU.

Uso:  python3 src/train.py --data data/locomotion.npz --epochs 200 --out weights/
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader, TensorDataset

from bimoe_model import BiMoE
from losses import total_loss
from preprocess import parse_bvh


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    bundle = np.load(args.data)
    inputs = torch.from_numpy(bundle["inputs"])
    targets = torch.from_numpy(bundle["targets"])
    train_end, val_end = int(bundle["train_end"]), int(bundle["val_end"])
    inferior_idx = sorted(set(int(j) for j in bundle["inferior_joints"]))
    superior_idx = sorted(set(int(j) for j in bundle["superior_joints"]))
    root_mean = torch.from_numpy(bundle["root_mean"].astype(np.float32))
    root_std = torch.from_numpy(bundle["root_std"].astype(np.float32))
    euler_min = torch.from_numpy(bundle["euler_min"])
    euler_max = torch.from_numpy(bundle["euler_max"])

    names, parents, offsets, _, _, _ = parse_bvh(SRC.parent / "dataset" / "additional_basic_locomotion_2.bvh")
    offsets_t = torch.tensor(np.array(offsets), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(inputs[:train_end], targets[:train_end]),
                              batch_size=args.batch, shuffle=True)
    val_data = (inputs[train_end:val_end], targets[train_end:val_end])

    model = BiMoE().to(device)
    print(f"Parametros: {model.count_parameters()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    patience_left = 20

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if args.noise > 0:
                # Robustez: ruido gaussiano en quats de entrada (aprende a volver).
                noisy = batch_x.clone()
                noisy[:, :276] += torch.randn_like(noisy[:, :276]) * args.noise
                batch_x = noisy
            lower, upper, _ = model(batch_x)
            loss, _ = total_loss(lower, upper, batch_y, offsets_t.to(device), parents,
                                 inferior_idx, superior_idx, root_mean.to(device),
                                 root_std.to(device), euler_min.to(device),
                                 euler_max.to(device), model=model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            lower, upper, _ = model(val_data[0].to(device))
            val_loss, parts = total_loss(lower, upper, val_data[1].to(device),
                                         offsets_t.to(device), parents, inferior_idx,
                                         superior_idx, root_mean.to(device),
                                         root_std.to(device), euler_min.to(device),
                                         euler_max.to(device))
        print(f"epoch {epoch:3d} train {train_loss / len(train_loader):.6f} "
              f"val {val_loss.item():.6f} (pose {parts['pose'].item():.6f} "
              f"foot {parts['foot'].item():.6f} smooth {parts['smooth'].item():.6f} "
              f"limit {parts['limit'].item():.6f})", flush=True)

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            patience_left = 20
            torch.save(model.state_dict(), out_dir / "bimoe10k_best.pt")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break
        if epoch % 10 == 0:
            torch.save(model.state_dict(), out_dir / f"bimoe10k_epoch{epoch}.pt")
    print(f"Mejor val: {best_val:.6f} -> {out_dir / 'bimoe10k_best.pt'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena BiMoE-10K")
    parser.add_argument("--data", default="data/locomotion.npz")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.01,
                        help="ruido gaussiano en quats de entrada (robustez)")
    parser.add_argument("--out", default="weights")
    return parser.parse_args()


if __name__ == "__main__":
    main()
