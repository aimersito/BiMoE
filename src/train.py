#!/usr/bin/env python3
"""Entrenamiento BiMoE-10K v4: residual + scheduled sampling + root supervisado.

v4 ataca el atractor congelado medido (jerk≈0 autoalimentado):
1. Salidas RESIDUALES (next = cur + delta): la salida constante deja de ser
   minimo trivial y la motion loss la castiga directo.
2. SCHEDULED SAMPLING con rampa 0->0.5: expone a estados propios utiles
   (100% self-fed desde epoca 1 alimentaba basura: pose 0.35).
3. ROOT con supervision directa (antes solo aprendia via FK).
4. Se conserva: BPTT truncado, clamps fisicos, guards NaN (0 skips en v3).

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


def unroll(model, state0, root_prev_raw, true_win, prev_true_win, true_root_win, offsets_t, parents,
           inferior_idx, superior_idx, root_mean, root_std, vel_mean, vel_std,
           feet_mean, feet_std, euler_min, euler_max, frame_time, w_motion,
           p_pred=1.0, detach_carry=True):
    """Desenrolla K pasos. v4: salidas RESIDUALES + SCHEDULED SAMPLING.
    p_pred = prob. de realimentar la prediccion propia (1-p = frame real GT).
    Rampa 0->~0.5 por epoca: al inicio estados propios serian basura.
    detach_carry=True = BPTT truncado (estable, bendecido por pushforward trick).
    true_root_win: (B,K,3) raices reales en metros (para sampling + loss root)."""
    n_joints = 23
    state = state0
    root_prev = root_prev_raw
    pose_sum = 0.0
    root_sum = 0.0
    foot_sum = 0.0
    limit_sum = 0.0
    motion_sum = 0.0
    stacked = []
    steps = true_win.shape[1]
    for k in range(steps):
        h0 = state[:, :92].reshape(-1, n_joints, 4)
        lower, upper, _ = model(state)
        full, root_raw, delta = assemble_prediction(lower, upper, h0, root_prev,
                                                    inferior_idx, superior_idx,
                                                    root_mean, root_std, n_joints)
        true_full = true_win[:, k, :92].reshape(-1, n_joints, 4)
        true_root = true_root_win[:, k]
        pose_sum = pose_sum + quaternion_sign_invariant_loss(full, true_full)
        root_sum = root_sum + (((root_raw - true_root) / root_std).pow(2)).mean()
        foot_sum = foot_sum + foot_sliding_loss(full, root_raw, offsets_t, parents,
                                                     (17, 21), (18, 22))
        limit_sum = limit_sum + joint_limit_loss(full, euler_min, euler_max)
        # Motion sobre deltas CRUDOS (el modelo ya es residual: delta directo).
        true_delta = true_full - prev_true_win[:, k]
        denom = true_delta.pow(2).mean().detach() + 1e-8
        motion_sum = motion_sum + torch.log1p((delta - true_delta).pow(2).mean() / denom)
        stacked.append(full)
        # Scheduled sampling: carry = mezcla propia/real segun p_pred.
        batch = state.shape[0]
        use_pred = (torch.rand(batch, 1, 1, device=state.device) < p_pred)
        carry_q = torch.where(use_pred, full.detach(), true_full)
        use_pred_r = use_pred.reshape(batch, 1)
        carry_root = torch.where(use_pred_r, root_raw.detach(), true_root)
        # Siguiente estado desde el carry (compuertas fisicas anti-NaN).
        root_raw_c = torch.clamp(carry_root, root_mean - 10.0, root_mean + 10.0)
        with torch.no_grad():
            pos = forward_kinematics_batch(carry_q, root_raw_c, offsets_t, parents)
        feet_raw = torch.stack([torch.minimum(pos[:, 21], pos[:, 22]),
                                torch.minimum(pos[:, 17], pos[:, 18])], dim=1)[:, :, 1]
        feet_raw_c = torch.clamp(feet_raw, -0.5, 2.0)
        root_n = (root_raw_c - root_mean) / root_std
        vel_raw = torch.clamp((root_raw_c - root_prev) / frame_time, -6.0, 6.0)
        vel_n = (vel_raw - vel_mean) / vel_std
        feet_n = (feet_raw_c - feet_mean) / feet_std
        state = torch.cat([carry_q.reshape(batch, -1),
                           state[:, :184], root_n, vel_n, feet_n], dim=1)
        root_prev = root_raw_c
        if detach_carry:
            state = state.detach()
            root_prev = root_prev.detach()
    stacked = torch.stack(stacked, dim=1)  # (B, K, J, 4)
    # Suavidad sobre el eje TIEMPO (dim=1), no sobre el batch.
    if steps >= 3:
        vel = torch.diff(stacked, dim=1)
        smooth = torch.diff(vel, dim=1).pow(2).mean()
    else:
        smooth = stacked.sum() * 0.0
    n = float(steps)
    return {"pose": pose_sum / n, "root": root_sum / n, "foot": foot_sum / n,
            "limit": limit_sum / n, "motion": motion_sum / n, "smooth": smooth,
            "stacked": stacked}


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
    val_true_root = val_true[:, :, 92:95] * root_std + root_mean  # metros

    model = BiMoE().to(device)
    print(f"Parametros: {model.count_parameters()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detach_carry = not args.full_bptt
    print(f"BPTT: {'completo (encadenado)' if args.full_bptt else 'truncado (estable)'}")
    best_val = float("inf")
    patience_left = 20
    dev = torch.device(device)
    off_d = offsets_t.to(dev)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        # Scheduled sampling: rampa lineal 0 -> ss_end (estados propios utiles,
        # nunca basura pura al inicio ni 100% autoalimentado con 1 clip).
        p_pred = args.ss_end * min(1.0, epoch / max(1, args.ss_ramp))
        for (starts,) in train_loader:
            state0 = inputs[starts].to(dev)
            if args.noise > 0:
                noisy = state0.clone()
                noisy[:, :276] += torch.randn_like(noisy[:, :276]) * args.noise
                state0 = noisy
            true_win, prev_win = build_windows(targets, starts, args.unroll)
            true_win, prev_win = true_win.to(dev), prev_win.to(dev)
            true_root_win = true_win[:, :, 92:95] * root_std.to(dev) + root_mean.to(dev)
            root_prev = (root_mean.to(dev) + targets[starts - 1][:, 92:95].to(dev) * root_std.to(dev))
            out = unroll(model, state0, root_prev, true_win, prev_win, true_root_win,
                         off_d, parents,
                         inferior_idx, superior_idx, root_mean.to(dev), root_std.to(dev),
                         vel_mean.to(dev), vel_std.to(dev), feet_mean.to(dev), feet_std.to(dev),
                         euler_min.to(dev), euler_max.to(dev), frame_time, args.w_motion,
                         p_pred=p_pred, detach_carry=detach_carry)
            reg = sum(p.pow(2).sum() for p in model.parameters())
            loss = (out["pose"] + out["root"] + 0.5 * out["foot"] + 0.3 * out["smooth"]
                    + 0.5 * out["limit"] + args.w_motion * out["motion"] + 1e-5 * reg)
            if not torch.isfinite(loss):
                with torch.no_grad():
                    diag = {k: (v.item() if torch.is_tensor(v) else v)
                            for k, v in {"pose": out["pose"], "root": out["root"],
                                         "foot": out["foot"],
                                         "smooth": out["smooth"], "limit": out["limit"],
                                         "motion": out["motion"]}.items()}
                print(f"  [epoch {epoch}] batch no-finito: {diag}", flush=True)
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Anti-envenenamiento: un grad inf (p.ej. asin en el borde x8 BPTT)
            # vuelve NaN el clip (inf*0) y mata los pesos para siempre.
            if not all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                print(f"  [epoch {epoch}] grad no-finito, se salta", flush=True)
                optimizer.zero_grad()
                continue
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            # Val con el MISMO p_pred final (mide el regimen del juego).
            v = unroll(model, val_state0.to(dev), val_root_prev.to(dev),
                       val_true.to(dev), val_prev.to(dev), val_true_root.to(dev),
                       off_d, parents,
                       inferior_idx, superior_idx, root_mean.to(dev), root_std.to(dev),
                       vel_mean.to(dev), vel_std.to(dev), feet_mean.to(dev), feet_std.to(dev),
                       euler_min.to(dev), euler_max.to(dev), frame_time, args.w_motion,
                       p_pred=args.ss_end)
            val_tensor = (v["pose"] + v["root"] + 0.5 * v["foot"] + 0.5 * v["limit"]
                          + args.w_motion * v["motion"])
            val_finite = bool(torch.isfinite(val_tensor).item())
            val_loss = val_tensor.item() if val_finite else float("nan")
            # Referencia one-step clasica sobre pares de val (base real -> target).
            lo, up, _ = model(val_state0[:512].to(dev))
            ref, _ = total_loss(lo, up, val_state0[:512].to(dev), val_true[:512, 0].to(dev),
                                off_d, parents,
                                inferior_idx, superior_idx, root_mean.to(dev),
                                root_std.to(dev), euler_min.to(dev), euler_max.to(dev))
        print(f"epoch {epoch:3d} train {train_loss / len(train_loader):.6f} "
              f"val_unroll {val_loss:.6f} (pose {v['pose'].item():.6f} "
              f"root {v['root'].item():.6f} motion {v['motion'].item():.6f} "
              f"foot {v['foot'].item():.6f} "
              f"limit {v['limit'].item():.6f}) ref1 {ref.item():.6f} p {p_pred:.2f}", flush=True)

        if val_finite and val_loss < best_val:
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
    parser.add_argument("--full-bptt", action="store_true",
                        help="gradiente encadenado entre pasos (por defecto: truncado)")
    parser.add_argument("--w_motion", type=float, default=0.15,
                        help="peso motion (guia sin ahogar a pose; v3 usaba 0.5)")
    parser.add_argument("--ss-end", type=float, default=0.5,
                        help="scheduled sampling final: prob. de realimentar prediccion")
    parser.add_argument("--ss-ramp", type=int, default=60,
                        help="epocas para subir el sampling de 0 a ss-end")
    parser.add_argument("--out", default="weights")
    return parser.parse_args()


if __name__ == "__main__":
    main()
