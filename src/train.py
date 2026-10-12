#!/usr/bin/env python3
"""Entrenamiento BiMoE-10K v3: desenrollado multi-paso + motion loss.

El entreno one-step puro crea un atractor congelado (jerk≈0 autoalimentado):
el modelo aprende a copiar el historial y en juego se clava en una pose fija
con los pies flotando. Aqui cada batch desenrolla K pasos alimentando las
PROPIAS predicciones (scheduled sampling 100%) y se castiga predecir delta
cero cuando el target se mueve. La seleccion (best.pt) usa val desenrollado,
que es la metrica del juego, no el one-step.

Uso:  python3 src/train.py --data data/locomotion.npz --epochs 200 --unroll 8 --out weights/
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
from losses import (assemble_prediction, foot_sliding_loss,
                    joint_limit_loss, quaternion_sign_invariant_loss,
                    forward_kinematics_batch, total_loss)
from preprocess import parse_bvh


def build_windows(targets, starts, steps):
    """Ventanas de targets: true_win[k] y prev_true_win[k] (quats del frame previo)."""
    idx = starts[:, None] + torch.arange(steps)[None, :]
    true_win = targets[idx]                       # (B, K, 95)
    prev_idx = starts[:, None] - 1 + torch.arange(steps)[None, :]
    prev_true = targets[prev_idx][:, :, :92].reshape(-1, steps, 23, 4)
    return true_win, prev_true


def unroll(model, state0, root_prev_raw, true_win, prev_true_win, offsets_t, parents,
           inferior_idx, superior_idx, root_mean, root_std, vel_mean, vel_std,
           feet_mean, feet_std, euler_min, euler_max, frame_time, w_motion):
    """Desenrolla K pasos con estados propios. Devuelve perdidas medias + quats apilados."""
    n_joints = 23
    state = state0
    root_prev = root_prev_raw
    pose_sum = 0.0
    foot_sum = 0.0
    limit_sum = 0.0
    motion_sum = 0.0
    stacked = []
    steps = true_win.shape[1]
    for k in range(steps):
        lower, upper, _ = model(state)
        full, root_raw = assemble_prediction(lower, upper, inferior_idx, superior_idx,
                                             root_mean, root_std, n_joints)
        true_full = true_win[:, k, :92].reshape(-1, n_joints, 4)
        pose_sum = pose_sum + quaternion_sign_invariant_loss(full, true_full)
        foot_sum = foot_sum + foot_sliding_loss(full, root_raw, offsets_t, parents,
                                                     (17, 21), (18, 22))
        limit_sum = limit_sum + joint_limit_loss(full, euler_min, euler_max)
        # Motion: castiga predecir delta cero cuando el target se mueve.
        # Relativo a la energia del delta real (adimensional; 1.0 = quedarse
        # quieto) en escala log1p para que no explote al inicio del entreno.
        h0 = state[:, :92].reshape(-1, n_joints, 4)
        pred_delta = full - h0
        true_delta = true_full - prev_true_win[:, k]
        denom = true_delta.pow(2).mean().detach() + 1e-8
        motion_sum = motion_sum + torch.log1p((pred_delta - true_delta).pow(2).mean() / denom)
        stacked.append(full)
        # Reconstruye el siguiente estado desde las predicciones (diferenciable).
        with torch.no_grad():
            pos = forward_kinematics_batch(full, root_raw, offsets_t, parents)
        feet_raw = torch.stack([torch.minimum(pos[:, 21], pos[:, 22]),
                                torch.minimum(pos[:, 17], pos[:, 18])], dim=1)[:, :, 1]
        root_n = (root_raw - root_mean) / root_std
        vel_n = ((root_raw - root_prev) / frame_time - vel_mean) / vel_std
        feet_n = (feet_raw - feet_mean) / feet_std
        state = torch.cat([full.reshape(full.shape[0], -1),
                           state[:, :184], root_n, vel_n, feet_n], dim=1)
        root_prev = root_raw
    stacked = torch.stack(stacked, dim=1)  # (B, K, J, 4)
    # Suavidad sobre el eje TIEMPO (dim=1), no sobre el batch.
    if steps >= 3:
        vel = torch.diff(stacked, dim=1)
        smooth = torch.diff(vel, dim=1).pow(2).mean()
    else:
        smooth = stacked.sum() * 0.0
    n = float(steps)
    return {"pose": pose_sum / n, "foot": foot_sum / n, "limit": limit_sum / n,
            "motion": motion_sum / n, "smooth": smooth, "stacked": stacked}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device} | unroll: {args.unroll} | w_motion: {args.w_motion}")

    bundle = np.load(args.data)
    inputs = torch.from_numpy(bundle["inputs"])
    targets = torch.from_numpy(bundle["targets"])
    train_end, val_end = int(bundle["train_end"]), int(bundle["val_end"])
    frame_time = float(bundle["frame_time"])
    inferior_idx = sorted(set(int(j) for j in bundle["inferior_joints"]))
    superior_idx = sorted(set(int(j) for j in bundle["superior_joints"]))
    root_mean = torch.from_numpy(bundle["root_mean"].astype(np.float32))
    root_std = torch.from_numpy(bundle["root_std"].astype(np.float32))
    vel_mean = torch.from_numpy(bundle["vel_mean"].astype(np.float32))
    vel_std = torch.from_numpy(bundle["vel_std"].astype(np.float32))
    feet_mean = torch.from_numpy(bundle["feet_mean"].astype(np.float32))
    feet_std = torch.from_numpy(bundle["feet_std"].astype(np.float32))
    euler_min = torch.from_numpy(bundle["euler_min"])
    euler_max = torch.from_numpy(bundle["euler_max"])

    names, parents, offsets, _, _, _ = parse_bvh(SRC.parent / "dataset" / "additional_basic_locomotion_2.bvh")
    offsets_t = torch.tensor(np.array(offsets), dtype=torch.float32)

    # Starts cronologicos (se necesita targets[starts-1] para el delta inicial).
    train_starts = torch.arange(1, train_end - args.unroll + 1)
    val_starts = torch.arange(train_end, val_end - args.unroll + 1)
    train_loader = DataLoader(TensorDataset(train_starts), batch_size=args.batch, shuffle=True)
    val_state0 = inputs[val_starts]
    val_true, val_prev = build_windows(targets, val_starts, args.unroll)
    val_root_prev = (root_mean + targets[val_starts - 1][:, 92:95] * root_std)

    model = BiMoE().to(device)
    print(f"Parametros: {model.count_parameters()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    patience_left = 20
    dev = torch.device(device)
    off_d = offsets_t.to(dev)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for (starts,) in train_loader:
            state0 = inputs[starts].to(dev)
            if args.noise > 0:
                noisy = state0.clone()
                noisy[:, :276] += torch.randn_like(noisy[:, :276]) * args.noise
                state0 = noisy
            true_win, prev_win = build_windows(targets, starts, args.unroll)
            true_win, prev_win = true_win.to(dev), prev_win.to(dev)
            root_prev = (root_mean.to(dev) + targets[starts - 1][:, 92:95].to(dev) * root_std.to(dev))
            out = unroll(model, state0, root_prev, true_win, prev_win, off_d, parents,
                         inferior_idx, superior_idx, root_mean.to(dev), root_std.to(dev),
                         vel_mean.to(dev), vel_std.to(dev), feet_mean.to(dev), feet_std.to(dev),
                         euler_min.to(dev), euler_max.to(dev), frame_time, args.w_motion)
            reg = sum(p.pow(2).sum() for p in model.parameters())
            loss = (out["pose"] + 0.5 * out["foot"] + 0.3 * out["smooth"]
                    + 0.5 * out["limit"] + args.w_motion * out["motion"] + 1e-5 * reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v = unroll(model, val_state0.to(dev), val_root_prev.to(dev),
                       val_true.to(dev), val_prev.to(dev), off_d, parents,
                       inferior_idx, superior_idx, root_mean.to(dev), root_std.to(dev),
                       vel_mean.to(dev), vel_std.to(dev), feet_mean.to(dev), feet_std.to(dev),
                       euler_min.to(dev), euler_max.to(dev), frame_time, args.w_motion)
            val_loss = (v["pose"] + 0.5 * v["foot"] + 0.5 * v["limit"]
                        + args.w_motion * v["motion"]).item()
            # Referencia one-step clasica sobre pares de val.
            lo, up, _ = model(val_state0[:512].to(dev))
            ref, _ = total_loss(lo, up, val_true[:512, 0].to(dev), off_d, parents,
                                inferior_idx, superior_idx, root_mean.to(dev),
                                root_std.to(dev), euler_min.to(dev), euler_max.to(dev))
        print(f"epoch {epoch:3d} train {train_loss / len(train_loader):.6f} "
              f"val_unroll {val_loss:.6f} (pose {v['pose'].item():.6f} "
              f"motion {v['motion'].item():.6f} foot {v['foot'].item():.6f} "
              f"limit {v['limit'].item():.6f}) ref1 {ref.item():.6f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            patience_left = 20
            torch.save(model.state_dict(), out_dir / "bimoe10k_best.pt")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break
        if epoch % 10 == 0:
            torch.save(model.state_dict(), out_dir / f"bimoe10k_epoch{epoch}.pt")
    print(f"Mejor val_unroll: {best_val:.6f} -> {out_dir / 'bimoe10k_best.pt'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena BiMoE-10K v3 (desenrollado)")
    parser.add_argument("--data", default="data/locomotion.npz")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.01,
                        help="ruido gaussiano en quats de entrada (robustez)")
    parser.add_argument("--unroll", type=int, default=8,
                        help="pasos autoalimentados por batch (anti-congelamiento)")
    parser.add_argument("--w_motion", type=float, default=0.5,
                        help="peso de la motion loss (castiga quedarse quieto)")
    parser.add_argument("--out", default="weights")
    return parser.parse_args()


if __name__ == "__main__":
    main()
