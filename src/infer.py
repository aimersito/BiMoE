#!/usr/bin/env python3
"""Inference BiMoE-10K: rollout autoregresivo desde un frame semilla.

Entrada con historial: [q_t, q_t-1, q_t-2] + root + vel + pies (284).
Salida: generated/rollout.npz + generated/rollout.bvh

Uso:  .venv/bin/python src/infer.py --frames 600
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import torch

from bimoe_model import BiMoE
from losses import forward_kinematics_batch
from preprocess import matrix_to_euler_yxz, parse_bvh


def main():
    args = parse_args()
    bundle = np.load(ROOT / "data" / "locomotion.npz")
    inputs = torch.from_numpy(bundle["inputs"])
    targets = torch.from_numpy(bundle["targets"])
    val_end = int(bundle["val_end"])
    train_end = int(bundle["train_end"])
    inferior_idx = sorted(set(int(j) for j in bundle["inferior_joints"]))
    superior_idx = sorted(set(int(j) for j in bundle["superior_joints"]))
    root_mean = bundle["root_mean"].astype(np.float32)
    root_std = bundle["root_std"].astype(np.float32)

    names, parents, offsets, channel_orders, motion, frame_time = parse_bvh(
        ROOT / "dataset" / "additional_basic_locomotion_2.bvh")
    offsets_t = torch.tensor(np.array(offsets), dtype=torch.float32)

    feet_mean = inputs[:train_end, 282:284].mean(dim=0)
    feet_std = inputs[:train_end, 282:284].std(dim=0) + 1e-8
    vel_mean = inputs[:train_end, 279:282].mean(dim=0)
    vel_std = inputs[:train_end, 279:282].std(dim=0) + 1e-8

    model = BiMoE()
    model.load_state_dict(torch.load(ROOT / "weights" / "bimoe10k_best.pt", map_location="cpu"))
    model.eval()
    print(f"Parametros: {model.count_parameters()}")

    # Frames reales: targets[i] es el frame (i+3).
    all_quats = targets[:, :92].reshape(-1, 23, 4).numpy()
    all_root = targets[:, 92:95].numpy() * root_std + root_mean

    seed = val_end  # primer target del test -> frame seed+3
    hist = [torch.from_numpy(all_quats[seed - 1]).float(),
            torch.from_numpy(all_quats[seed - 2]).float(),
            torch.from_numpy(all_quats[seed - 3]).float()]
    current_root = all_root[seed - 1].copy()
    prev_root = all_root[seed - 2].copy()
    real_quats = all_quats[seed:seed + args.frames]
    real_root = all_root[seed:seed + args.frames]

    gen_quats, gen_root = [], []
    with torch.no_grad():
        for _ in range(args.frames):
            root_t = torch.from_numpy(current_root).float()
            vel = (torch.from_numpy(current_root - prev_root).float() / frame_time
                   - vel_mean) / vel_std
            pos = forward_kinematics_batch(hist[0].reshape(1, 23, 4).float(),
                                           root_t.reshape(1, 3), offsets_t, parents)
            feet = torch.stack([torch.minimum(pos[0, 21], pos[0, 22]),
                                torch.minimum(pos[0, 17], pos[0, 18])])[:, 1]
            feet_n = (feet - feet_mean) / feet_std
            state = torch.cat([h.reshape(-1) for h in hist] +
                              [(root_t - torch.from_numpy(root_mean)) / torch.from_numpy(root_std),
                               vel, feet_n])
            lower, upper, _ = model(state)
            full = torch.zeros(23, 4)
            full[inferior_idx] = lower[:36].reshape(-1, 4)
            full[superior_idx] = upper.reshape(-1, 4)
            full = full / full.norm(dim=-1, keepdim=True)
            prev_root = current_root.copy()
            current_root = lower[36:39].numpy() * root_std + root_mean
            hist = [full, hist[0], hist[1]]
            gen_quats.append(full.numpy())
            gen_root.append(current_root.copy())

    gen_quats = np.array(gen_quats)
    gen_root = np.array(gen_root)
    count = min(len(gen_quats), len(real_quats))
    quat_error = np.minimum(((gen_quats[:count] - real_quats[:count]) ** 2).sum(axis=-1),
                            ((gen_quats[:count] + real_quats[:count]) ** 2).sum(axis=-1)).mean()
    root_error = np.sqrt(((gen_root[:count] - real_root[:count]) ** 2).sum(axis=-1)).mean()
    print(f"frames: {count} | quat_mse: {quat_error:.6f} | root_rmse_m: {root_error:.4f}")

    out_dir = ROOT / "generated"
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(out_dir / "rollout.npz", gen_quats=gen_quats.astype(np.float32),
                        gen_root=gen_root.astype(np.float32),
                        real_quats=real_quats[:count].astype(np.float32),
                        real_root=real_root[:count].astype(np.float32),
                        joint_names=bundle["joint_names"], frame_time=frame_time)
    write_bvh(out_dir / "rollout.bvh", names, parents, offsets, channel_orders,
              gen_quats, gen_root, frame_time)
    print(f"OK: {out_dir / 'rollout.npz'} y {out_dir / 'rollout.bvh'}")


def write_bvh(path, names, parents, offsets, channel_orders, quats, root, frame_time):
    """Escribe BVH con la jerarquia original y el movimiento generado."""
    lines = ["HIERARCHY"]
    children = {i: [] for i in range(len(names))}
    for i, p in enumerate(parents):
        if p != -1:
            children[p].append(i)

    def dump(joint):
        out = []
        head = "ROOT" if parents[joint] == -1 else "JOINT"
        out.append(f"{head} {names[joint]}")
        out.append("{")
        out.append(f"OFFSET {' '.join(f'{v:.6f}' for v in offsets[joint])}")
        channels = channel_orders[joint]
        out.append(f"CHANNELS {len(channels)} {' '.join(channels)}")
        for child in children[joint]:
            out.extend(dump(child))
        if not children[joint]:
            out.append("End Site")
            out.append("{")
            out.append("OFFSET 0.000000 0.000000 0.000000")
            out.append("}")
        out.append("}")
        return out

    lines.extend(dump(0))
    lines.append("MOTION")
    lines.append(f"Frames: {len(quats)}")
    lines.append(f"Frame Time: {frame_time:.6f}")
    for frame in range(len(quats)):
        values = []
        for joint in range(len(names)):
            channels = channel_orders[joint]
            if parents[joint] == -1:
                values.extend(f"{v:.6f}" for v in root[frame])
            x, y, z, w = quats[frame, joint]
            rot = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            ey, ex, ez = matrix_to_euler_yxz(rot)
            for channel in channels:
                if "position" in channel:
                    continue
                values.append(f"{ex:.6f}" if channel == "Xrotation" else
                              f"{ey:.6f}" if channel == "Yrotation" else f"{ez:.6f}")
        lines.append(" ".join(values))
    Path(path).write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Rollout BiMoE-10K")
    parser.add_argument("--frames", type=int, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    main()
