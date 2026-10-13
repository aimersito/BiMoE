#!/usr/bin/env python3
"""Build locomotion_v2.npz: 50 clips EMM -> pares (estado_t -> estado_t+1).

Reglas multi-clip:
- Ventanas NUNCA cruzan de clip (pares por clip, clip_id por par).
- Concatenacion ordenada [train | val | test] para que train_end/val_end
  sigan funcionando sin tocar el loop (solo se filtra por clip_id).
- Fase y periodo POR CLIP (cada ritmo es suyo); scalar phase_period =
  clip de referencia (compatibilidad con herramientas v1).
- Excluidos: *30f* (duplicado a 30fps), *tpose* (1 frame).

Split por clips (estratificado viaje/sitio):
  test: additional_basic_locomotion_2 (referencia v1), crouching, o4_3
  val:  basic_locomotion_2, circles_3, locomotion_loops_1, o1_4
  train: resto.

Uso: .venv/bin/python tools/build_v2.py [--src /home/luis/Escritorio/emm]
     Verificacion de paridad v1: --only additional_basic_locomotion_2
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from preprocess import (  # noqa: E402
    parse_bvh, euler_to_quaternion, quaternion_to_matrix,
    matrix_to_euler_yxz, forward_kinematics,
    INFERIOR_JOINTS, SUPERIOR_JOINTS, ANKLE_LEFT, ANKLE_RIGHT,
    TOE_LEFT, TOE_RIGHT,
)

TEST_CLIPS = ["additional_basic_locomotion_2", "crouching", "o4_3"]
VAL_CLIPS = ["basic_locomotion_2", "circles_3", "locomotion_loops_1", "o1_4"]


def gait_phase(foot_heights, frame_time):
    """Mismo algoritmo que preprocess v1, pero por clip. Devuelve (phase, period)."""
    sig = (foot_heights[:, 0] - foot_heights[:, 1]).astype(np.float64)
    sig = sig - sig.mean()
    rspec = np.fft.rfft(sig)
    rfreqs = np.fft.rfftfreq(len(sig), d=frame_time)
    power = np.abs(rspec) * ((rfreqs > 0.4) & (rfreqs < 3.0))
    f0 = rfreqs[int(np.argmax(power))]
    X = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), d=frame_time)
    keep = (np.abs(freqs - f0) < 0.25) | (np.abs(freqs + f0) < 0.25)
    H = np.zeros(len(sig))
    H[0] = 1.0
    if len(sig) % 2 == 0:
        H[1:len(sig) // 2] = 2.0
        H[len(sig) // 2] = 1.0
    else:
        H[1:(len(sig) + 1) // 2] = 2.0
    analytic = np.fft.ifft(X * keep * H)
    phase = (np.angle(analytic) / (2.0 * np.pi)) % 1.0
    unwrapped = np.unwrap(np.angle(analytic))
    step = np.diff(unwrapped)
    period = float(2.0 * np.pi / abs(step.mean())) if np.isfinite(step.mean()) and abs(step.mean()) > 1e-9 else 0.0
    if not np.isfinite(period) or period < 8 or period > len(sig):
        ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
        ac = ac / (ac[0] + 1e-12)
        period = 48.0
        for lag in range(10, 140):
            if ac[lag] > ac[lag - 1] and ac[lag] > ac[lag + 1] and ac[lag] > 0.3:
                period = float(lag)
                break
    return phase.astype(np.float32), float(period), float(f0)


def process_clip(path, ref_key):
    """Parsea un clip y devuelve frames procesados. Falla si el esqueleto difiere."""
    names, parents, offsets, channel_orders, data, frame_time = parse_bvh(str(path))
    assert len(names) == 23, f"{path.name}: {len(names)} nodos"
    key = (tuple(names), tuple(parents), tuple(map(tuple, offsets)),
           tuple(map(tuple, channel_orders)))
    assert key == ref_key, f"{path.name}: jerarquia distinta a la referencia"
    column = 0
    node_columns = []
    for channels in channel_orders:
        node_columns.append(list(range(column, column + len(channels))))
        column += len(channels)
    root_position = data[:, node_columns[0][:3]]
    n_frames = data.shape[0]
    quaternions = np.zeros((n_frames, 23, 4))
    for joint in range(23):
        cols = node_columns[joint]
        channels = channel_orders[joint]
        rotation_cols = [c for c, name in zip(cols, channels) if "rotation" in name]
        rotation_names = [name for name in channels if "rotation" in name]
        for frame in range(n_frames):
            quaternions[frame, joint] = euler_to_quaternion(data[frame, rotation_cols], rotation_names)
    for joint in range(23):
        for frame in range(1, n_frames):
            if np.dot(quaternions[frame - 1, joint], quaternions[frame, joint]) < 0.0:
                quaternions[frame, joint] *= -1.0
    foot_heights = np.zeros((n_frames, 2))
    for frame in range(n_frames):
        positions = forward_kinematics(quaternions[frame], root_position[frame], offsets, parents)
        foot_heights[frame, 0] = min(positions[ANKLE_LEFT, 1], positions[TOE_LEFT, 1])
        foot_heights[frame, 1] = min(positions[ANKLE_RIGHT, 1], positions[TOE_RIGHT, 1])
    root_velocity = np.diff(root_position, axis=0, prepend=root_position[:1]) / frame_time
    phase, period, f0 = gait_phase(foot_heights, frame_time)
    euler_all = np.zeros((n_frames, 23, 3))
    for frame in range(n_frames):
        for joint in range(23):
            euler_all[frame, joint] = matrix_to_euler_yxz(quaternion_to_matrix(quaternions[frame, joint]))
    return {"name": path.stem, "quats": quaternions, "root": root_position,
            "vel": root_velocity, "feet": foot_heights, "phase": phase,
            "period": period, "f0": f0, "euler": euler_all,
            "frame_time": frame_time, "names": names, "parents": parents,
            "offsets": offsets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/luis/Escritorio/emm")
    ap.add_argument("--out", default="data/locomotion_v2.npz")
    ap.add_argument("--only", default=None, help="paridad v1: procesar un solo clip")
    args = ap.parse_args()

    files = sorted(Path(args.src).glob("*.bvh"))
    files = [f for f in files if "30f" not in f.stem and "tpose" not in f.stem]
    if args.only:
        files = [f for f in files if f.stem == args.only]
        assert files, f"clip {args.only} no encontrado"
    print(f"clips: {len(files)}")

    # Referencia = clip actual v1 (jerarquia que el resto debe igualar).
    ref_path = Path(__file__).resolve().parent.parent / "dataset" / "additional_basic_locomotion_2.bvh"
    rn, rp, ro, rc, _, _ = parse_bvh(str(ref_path))
    ref_key = (tuple(rn), tuple(rp), tuple(map(tuple, ro)), tuple(map(tuple, rc)))

    clips = []
    for f in files:
        c = process_clip(f, ref_key)
        assert abs(c["frame_time"] - 0.016667) < 1e-6, f'{f.name}: dt {c["frame_time"]}'
        clips.append(c)
        print(f"  {c['name']:38s} frames {c['quats'].shape[0]:5d} "
              f"periodo {c['period']:5.1f} f0 {c['f0']:.2f}Hz")

    def split_of(name):
        if args.only:
            return "train"
        if name in TEST_CLIPS:
            return "test"
        if name in VAL_CLIPS:
            return "val"
        return "train"

    order = {"train": 0, "val": 1, "test": 2}
    clips.sort(key=lambda c: (order[split_of(c["name"])], c["name"]))

    # Stats globales (convencion v1: todo el corpus, incluye test).
    all_root = np.concatenate([c["root"] for c in clips], axis=0)
    all_vel = np.concatenate([c["vel"] for c in clips], axis=0)
    all_feet = np.concatenate([c["feet"] for c in clips], axis=0)
    root_mean, root_std = all_root.mean(axis=0), all_root.std(axis=0) + 1e-8
    vel_mean, vel_std = all_vel.mean(axis=0), all_vel.std(axis=0) + 1e-8
    feet_mean, feet_std = all_feet.mean(axis=0), all_feet.std(axis=0) + 1e-8
    euler_min = np.concatenate([c["euler"].min(axis=0, keepdims=True) for c in clips], axis=0).min(axis=0).astype(np.float32)
    euler_max = np.concatenate([c["euler"].max(axis=0, keepdims=True) for c in clips], axis=0).max(axis=0).astype(np.float32)

    # Pares por clip (historial dentro del clip, nunca cruza).
    ins, tgs, cids, phases = [], [], [], []
    bounds, names, periods, splits = [], [], [], []
    cursor = 0
    for ci, c in enumerate(clips):
        n = c["quats"].shape[0]
        flat = c["quats"].reshape(n, -1)
        rn_ = (c["root"] - root_mean) / root_std
        vn = (c["vel"] - vel_mean) / vel_std
        fn = (c["feet"] - feet_mean) / feet_std
        inp = np.concatenate([flat[2:-1], flat[1:-2], flat[:-3],
                              rn_[2:-1], vn[2:-1], fn[2:-1],
                              c["phase"][2:-1, None]], axis=1)
        tgt = np.concatenate([flat[3:], rn_[3:]], axis=1)
        ins.append(inp)
        tgs.append(tgt)
        cids.append(np.full(inp.shape[0], ci, dtype=np.int32))
        phases.append(c["phase"])
        bounds.append((cursor, cursor + inp.shape[0]))
        cursor += inp.shape[0]
        names.append(c["name"])
        periods.append(c["period"])
        splits.append(split_of(c["name"]))

    inputs = np.concatenate(ins, axis=0).astype(np.float32)
    targets = np.concatenate(tgs, axis=0).astype(np.float32)
    clip_id = np.concatenate(cids, axis=0)
    phase_all = np.concatenate(phases, axis=0).astype(np.float32)

    if args.only:
        train_end, val_end = int(len(inputs) * 0.8), int(len(inputs) * 0.9)
    else:
        train_end = sum(e - s for (s, e), sp in zip(bounds, splits) if sp == "train")
        # Los bloques estan ordenados train|val|test: el corte es exacto.
        val_end = train_end + sum(e - s for (s, e), sp in zip(bounds, splits) if sp == "val")

    ref_period = dict(zip(names, periods)).get(
        "additional_basic_locomotion_2", float(np.median(periods)))
    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        inputs=inputs, targets=targets,
        train_end=train_end, val_end=val_end,
        joint_names=np.array(clips[0]["names"]),
        inferior_joints=np.array(INFERIOR_JOINTS),
        superior_joints=np.array(SUPERIOR_JOINTS),
        euler_min=euler_min, euler_max=euler_max,
        root_mean=root_mean, root_std=root_std,
        vel_mean=vel_mean, vel_std=vel_std,
        feet_mean=feet_mean, feet_std=feet_std,
        phase=phase_all, phase_period=np.float32(ref_period),
        phase_periods=np.array(periods, dtype=np.float32),
        clip_id=clip_id, clip_names=np.array(names),
        clip_bounds=np.array(bounds, dtype=np.int64),
        clip_splits=np.array(splits),
        frame_time=0.016667,
    )
    n_test = len(inputs) - val_end
    print(f"OK: {out}")
    print(f"pares {len(inputs)} | train {train_end} val {val_end - train_end} test {n_test}")
    print(f"periodo ref {ref_period:.1f} | clips train/val/test: "
          f"{splits.count('train')}/{splits.count('val')}/{splits.count('test')}")


if __name__ == "__main__":
    main()
