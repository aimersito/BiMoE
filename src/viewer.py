#!/usr/bin/env python3
"""Visor BiMoE-10K (Open3D): real (azul) vs generado (rojo), lado a lado.

Uso:  .venv/bin/python src/viewer.py [--frames 600]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import open3d as o3d

from preprocess import forward_kinematics, parse_bvh


def skeleton_lines(positions, parents):
    """Puntos + lineas del esqueleto a partir de posiciones mundo."""
    points = [positions[i] for i in range(len(positions))]
    lines = [[p, i] for i, p in enumerate(parents) if p != -1]
    return points, lines


def main():
    args = parse_args()
    rollout = np.load(ROOT / "generated" / "rollout.npz")
    names, parents, offsets, _, _, frame_time = parse_bvh(
        ROOT / "dataset" / "additional_basic_locomotion_2.bvh")
    frames = min(args.frames, len(rollout["gen_quats"]))
    shift = np.array([3.0, 0.0, 0.0])  # generado a la derecha del real

    vis = o3d.visualization.Visualizer()
    vis.create_window("BiMoE-10K: real (azul) vs generado (rojo)")
    real_cloud = o3d.geometry.PointCloud()
    gen_cloud = o3d.geometry.PointCloud()
    real_lines = o3d.geometry.LineSet()
    gen_lines = o3d.geometry.LineSet()
    vis.add_geometry(real_cloud)
    vis.add_geometry(gen_cloud)
    vis.add_geometry(real_lines)
    vis.add_geometry(gen_lines)

    for frame in range(frames):
        real_pos = forward_kinematics(rollout["real_quats"][frame], rollout["real_root"][frame],
                                      offsets, parents)
        gen_pos = forward_kinematics(rollout["gen_quats"][frame], rollout["gen_root"][frame],
                                     offsets, parents) + shift
        real_cloud.points = o3d.utility.Vector3dVector(real_pos)
        real_cloud.colors = o3d.utility.Vector3dVector(np.tile([0.2, 0.4, 1.0], (len(names), 1)))
        gen_cloud.points = o3d.utility.Vector3dVector(gen_pos)
        gen_cloud.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.3, 0.2], (len(names), 1)))
        edge = [[p, i] for i, p in enumerate(parents) if p != -1]
        real_lines.points = o3d.utility.Vector3dVector(real_pos)
        real_lines.lines = o3d.utility.Vector2iVector(edge)
        gen_lines.points = o3d.utility.Vector3dVector(gen_pos)
        gen_lines.lines = o3d.utility.Vector2iVector(edge)
        vis.update_geometry(real_cloud)
        vis.update_geometry(gen_cloud)
        vis.update_geometry(real_lines)
        vis.update_geometry(gen_lines)
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser(description="Visor BiMoE-10K")
    parser.add_argument("--frames", type=int, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    main()
