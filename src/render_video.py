#!/usr/bin/env python3
"""Renderiza rollout v4 a mp4 (stick figure 3D, liviano para PC debil).

Uso: .venv/bin/python src/render_video.py --weights weights/bimoe10k_best.pt --frames 120 --out /tmp/opencode/bimoe_v4.mp4
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import torch

from bimoe_model import BiMoE
from mono_model import MonoNet, MonoFiLM, RootNet, predict_next
from losses import forward_kinematics_batch
from preprocess import parse_bvh


def main():
    a = parse_args()
    bundle = np.load(ROOT / (a.data or "data/locomotion.npz"))
    names, parents, offsets, _, _, frame_time = parse_bvh(
        ROOT / "dataset" / "additional_basic_locomotion_2.bvh")
    ft = float(frame_time)
    ckpt = torch.load(a.weights, map_location="cpu")
    # Formato nuevo (dict con model_state_dict) o legacy (solo pesos).
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print(f"Checkpoint: epoch {ckpt.get('epoch')} best {ckpt.get('best_val')}")
    # Autodeteccion: keys con 'router' = MoE, con 'film' = mono-FiLM, si no = mono (misma regla que eval_gates).
    arch = "moe" if any("router" in k for k in state.keys()) else "mono"
    h = int((ckpt.get("config", {}) or {}).get("hidden", 56)) if isinstance(ckpt, dict) else 56
    model = BiMoE() if arch == "moe" else (MonoFiLM(hidden=h) if any("film" in k for k in state.keys()) else MonoNet(hidden=h))
    model.load_state_dict(state)
    model.eval()
    print(f"Arquitectura detectada: {arch}")
    root_net = None
    if isinstance(ckpt, dict) and ckpt.get("root_state_dict") is not None and arch == "mono":
        cfg = ckpt.get("config", {}) or {}
        root_net = RootNet(hidden=int(cfg.get("root_net", 32)))
        root_net.load_state_dict(ckpt["root_state_dict"])
        root_net.eval()
        print(f"Cabeza root aislada: {root_net.count_parameters()} params")
    reparam, dq_std, delta_std = False, None, None
    if isinstance(ckpt, dict) and (ckpt.get("config", {}) or {}).get("reparam") == "dq":
        reparam = True
        ins = bundle["inputs"][:int(bundle["train_end"]), :276].reshape(-1, 3, 92)
        dq = (ins[:, 0] - ins[:, 1])
        dq_std = torch.from_numpy(dq.std(axis=0, keepdims=True).astype(np.float32)).clamp_min(1e-4)
        dseq = bundle["targets"][:int(bundle["train_end"]), :92]
        delta_std = torch.from_numpy(
            (dseq[1:] - dseq[:-1]).std(axis=0, keepdims=True).astype(np.float32)).clamp_min(1e-4)
        print("Reparam dq activa en este ckpt")
    st = {k: torch.from_numpy(bundle[k].astype(np.float32))
          for k in ["root_mean", "root_std", "vel_mean", "vel_std", "feet_mean", "feet_std"]}
    rm, rs, vm, vs, fm, fs = st["root_mean"], st["root_std"], st["vel_mean"], st["vel_std"], st["feet_mean"], st["feet_std"]
    tq = torch.from_numpy(bundle["targets"][:, :92]).reshape(-1, 23, 4).float()
    tr = (torch.from_numpy(bundle["targets"][:, 92:95].astype(np.float32)) * rs + rm).numpy().astype(np.float64)
    off = torch.tensor(np.array(offsets), dtype=torch.float32)
    inf_idx = sorted(set(int(j) for j in bundle["inferior_joints"]))
    sup_idx = sorted(set(int(j) for j in bundle["superior_joints"]))
    n = len(tq)
    if "clip_id" in bundle:  # multi-clip: seed en el primer clip de test con hueco
        bounds = np.asarray(bundle["clip_bounds"])
        va = int(bundle["val_end"])
        pick = [i for i in range(len(bounds)) if bounds[i, 0] >= va
                and bounds[i, 1] - bounds[i, 0] >= a.frames + 10][0]
        lo = int(bounds[pick, 0])
        seed = lo + 10
        seed_lim = int(bounds[pick, 1]) - 1
        period = float(np.asarray(bundle["phase_periods"])[int(np.asarray(bundle["clip_id"])[lo])])
    else:
        seed = min(int(bundle["val_end"]) + 10, n - a.frames - 2)
        seed_lim = n - 1
        period = float(bundle["phase_period"])
    hist = [tq[seed].clone(), tq[seed - 1].clone(), tq[seed - 2].clone()]
    root = tr[seed].copy()
    ph = float(bundle["phase"][seed + 3] % 1.0)
    true_vel = (tr[1:] - tr[:-1]) / ft
    quats, roots = [], []
    with torch.no_grad():
        for h in range(a.frames):
            v = true_vel[min(seed + h, seed_lim - 1)]
            root = root + v * ft
            root_t = torch.from_numpy(root).float()
            vel_t = torch.from_numpy(v).float()
            pos = forward_kinematics_batch(hist[0].reshape(1, 23, 4).float(),
                                           root_t.reshape(1, 3), off, parents)
            feet = torch.stack([torch.minimum(pos[0, 21], pos[0, 22]),
                                torch.minimum(pos[0, 17], pos[0, 18])])[:, 1]
            st = torch.cat([x.reshape(-1) for x in hist] +
                              [(root_t - rm) / rs, (vel_t - vm) / vs, (feet - fm) / fs,
                               torch.tensor([ph])])
            if reparam:
                q0, q1, q2 = st[:92], st[92:184], st[184:276]
                st = torch.cat([q0, (q0 - q1) / dq_std.squeeze(0),
                                (q1 - q2) / dq_std.squeeze(0), st[276:]])
            if arch == "mono":
                d95 = model(st.unsqueeze(0))
                if delta_std is not None:
                    d95 = torch.cat([d95[:, :92] * delta_std, d95[:, 92:95]], dim=1)
                if root_net is not None:
                    d95 = torch.cat([d95[:, :92], root_net(st.unsqueeze(0))], dim=1)
                nxt, _ = predict_next(d95,
                                      hist[0].reshape(1, 23, 4),
                                      root_t.reshape(1, 3), rs)
                nxt = nxt.reshape(23, 4)
            else:
                lower, upper, _ = model(state)
                d = torch.zeros(23, 4)
                d[inf_idx] = lower[:36].reshape(-1, 4)
                d[sup_idx] = upper.reshape(-1, 4)
                nxt = hist[0] + d
                nxt = nxt / nxt.norm(dim=-1, keepdim=True).clamp_min(1e-4)
            hist = [nxt, hist[0], hist[1]]
            ph = (ph + 1.0 / period) % 1.0
            quats.append(nxt.numpy())
            roots.append(root.copy())
    quats = torch.from_numpy(np.array(quats)).float()
    roots = torch.from_numpy(np.array(roots)).float()
    with torch.no_grad():
        joints = forward_kinematics_batch(quats, roots, off, parents).numpy()  # (F,23,3)
    bones = [(i, p) for i, p in enumerate(parents) if p >= 0]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    c = joints.reshape(-1, 3)
    mid = np.median(c, axis=0)
    r = 0.9
    writer = FFMpegWriter(fps=60, codec="libx264")
    with writer.saving(fig, str(out), dpi=80):
        for f in range(0, len(joints), 2):  # cada 2 = 60f en 2s, liviano
            ax.clear()
            P = joints[f]
            for i, p in bones:
                xs = [P[p, 0], P[i, 0]]
                ys = [P[p, 2], P[i, 2]]
                zs = [P[p, 1], P[i, 1]]
                ax.plot(xs, ys, zs, "b-")
            ax.scatter(P[:, 0], P[:, 2], P[:, 1], s=6, c="r")
            ax.set_xlim(mid[0] - r, mid[0] + r)
            ax.set_ylim(mid[2] - r, mid[2] + r)
            ax.set_zlim(0, 1.9)
            ax.set_title(f"BiMoE rollout f{f} ({arch})")
            ax.view_init(elev=12, azim=-60)
            writer.grab_frame()
    print(f"VIDEO-OK {out} ({out.stat().st_size/1024:.0f} KB)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/bimoe10k_best.pt")
    p.add_argument("--data", default="data/locomotion.npz")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--out", default="/tmp/opencode/bimoe_v4.mp4")
    return p.parse_args()


if __name__ == "__main__":
    main()
