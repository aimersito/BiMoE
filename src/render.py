#!/usr/bin/env python3
"""Render BiMoE-10K a video (matplotlib 3D): real (azul) vs generado (rojo).

Uso:  .venv/bin/python src/render.py [--frames 600] [--stride 2]
Salida: generated/rollout.mp4 + generated/contacto.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from preprocess import forward_kinematics, parse_bvh


def compute_positions(quats, roots, offsets, parents):
    return np.array([forward_kinematics(quats[f], roots[f], offsets, parents)
                     for f in range(len(quats))])


def draw_skeleton(axis, positions, parents, color):
    for i, parent in enumerate(parents):
        if parent == -1:
            continue
        segment = np.array([positions[parent], positions[i]])
        axis.plot(segment[:, 0], segment[:, 2], segment[:, 1], c=color, linewidth=2)
    axis.scatter(positions[:, 0], positions[:, 2], positions[:, 1], c=color, s=8)


def main():
    args = parse_args()
    rollout = np.load(ROOT / "generated" / "rollout.npz")
    names, parents, offsets, _, _, frame_time = parse_bvh(
        ROOT / "dataset" / "additional_basic_locomotion_2.bvh")
    total = min(args.frames, len(rollout["gen_quats"]))
    print("FK del rollout ...")
    real = compute_positions(rollout["real_quats"][:total], rollout["real_root"][:total],
                             offsets, parents)
    gen = compute_positions(rollout["gen_quats"][:total], rollout["gen_root"][:total],
                            offsets, parents)
    shift = np.array([3.0, 0.0, 0.0])
    gen = gen + shift

    writer = FFMpegWriter(fps=int(1 / frame_time / args.stride), bitrate=2000)
    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(111, projection="3d")
    frames = range(0, total, args.stride)
    out_video = ROOT / "generated" / "rollout.mp4"
    with writer.saving(figure, str(out_video), dpi=100):
        for count, frame in enumerate(frames):
            axis.clear()
            draw_skeleton(axis, real[frame], parents, "blue")
            draw_skeleton(axis, gen[frame], parents, "red")
            axis.set_title(f"frame {frame}  azul=real  rojo=BiMoE-10K")
            axis.set_xlim(-2, 5)
            axis.set_ylim(-2, 2)
            axis.set_zlim(0, 2)
            axis.set_xlabel("X")
            axis.set_ylabel("Z")
            writer.grab_frame()
            if count % 50 == 0:
                print(f"  frame {frame}/{total}")
    print(f"OK: {out_video}")

    # Hoja de contacto: 6 instantes del generado.
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), subplot_kw={"projection": "3d"})
    for ax, frame in zip(axes.flat, np.linspace(0, total - 1, 6, dtype=int)):
        draw_skeleton(ax, gen[frame], parents, "red")
        ax.set_title(f"frame {frame}")
        ax.set_xlim(1, 5)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 2)
    out_img = ROOT / "generated" / "contacto.png"
    figure.tight_layout()
    figure.savefig(out_img, dpi=100)
    print(f"OK: {out_img}")


def parse_args():
    parser = argparse.ArgumentParser(description="Render BiMoE-10K")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--stride", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
