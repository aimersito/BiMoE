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
from losses import forward_kinematics_batch
from preprocess import parse_bvh


def main():
    a = parse_args()
    bundle = np.load(ROOT / "data" / "locomotion.npz")
    names, parents, offsets, _, _, frame_time = parse_bvh(
        ROOT / "dataset" / "additional_basic_locomotion_2.bvh")
    ft = float(frame_time)
    model = BiMoE()
    model.load_state_dict(torch.load(a.weights, map_location="cpu"))
    model.eval()
    st = {k: torch.from_numpy(bundle[k].astype(np.float32))
          for k in ["root_mean", "root_std", "vel_mean", "vel_std", "feet_mean", "feet_std"]}
    rm, rs, vm, vs, fm, fs = st["root_mean"], st["root_std"], st["vel_mean"], st["vel_std"], st["feet_mean"], st["feet_std"]
    tq = torch.from_numpy(bundle["targets"][:, :92]).reshape(-1, 23, 4).float()
    tr = (torch.from_numpy(bundle["targets"][:, 92:95].astype(np.float32)) * rs + rm).numpy().astype(np.float64)
    off = torch.tensor(np.array(offsets), dtype=torch.float32)
    inf_idx = sorted(set(int(j) for j in bundle["inferior_joints"]))
    sup_idx = sorted(set(int(j) for j in bundle["superior_joints"]))
    n = len(tq)
    seed = min(int(bundle["val_end"]) + 10, n - a.frames - 2)
    hist = [tq[seed].clone(), tq[seed - 1].clone(), tq[seed - 2].clone()]
    root = tr[seed].copy()
    ph = float(bundle["phase"][seed + 3] % 1.0)
    period = float(bundle["phase_period"])
    true_vel = (tr[1:] - tr[:-1]) / ft
    quats, roots = [], []
    with torch.no_grad():
        for h in range(a.frames):
            v = true_vel[min(seed + h, len(true_vel) - 1)]
            root = root + v * ft
            root_t = torch.from_numpy(root).float()
            vel_t = torch.from_numpy(v).float()
            pos = forward_kinematics_batch(hist[0].reshape(1, 23, 4).float(),
                                           root_t.reshape(1, 3), off, parents)
            feet = torch.stack([torch.minimum(pos[0, 21], pos[0, 22]),
                                torch.minimum(pos[0, 17], pos[0, 18])])[:, 1]
            state = torch.cat([x.reshape(-1) for x in hist] +
                              [(root_t - rm) / rs, (vel_t - vm) / vs, (feet - fm) / fs,
                               torch.tensor([ph])])
            lower, upper, _ = model(state)
            d = torch.zeros(23, 4)
            d[inf_idx] = lower[:36].reshape(-1, 4)
            d[sup_idx] = upper.reshape(-1, 4)
            nxt = hist[0] + d
            nxt = nxt / nxt.norm(dim=-1, keepdim=True)
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
            ax.set_title(f"BiMoE v4 rollout f{f}")
            ax.view_init(elev=12, azim=-60)
            writer.grab_frame()
    print(f"VIDEO-OK {out} ({out.stat().st_size/1024:.0f} KB)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/bimoe10k_best.pt")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--out", default="/tmp/opencode/bimoe_v4.mp4")
    return p.parse_args()


if __name__ == "__main__":
    main()
