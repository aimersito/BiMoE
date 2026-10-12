#!/usr/bin/env python3
"""Auditoria de gradientes por termino (analisis §3): magnitud vs direccion.

No mires losses, mira gradientes del tronco compartido:
  r = ||push||/||foot|| (push = pose+root), cosenos foot-vs-push/pose/root.
  r_med >10 = magnitud (foot ahogado -> apuesta 5 o w_foot x r/3, PCGrad no).
  frac_neg >0.4 y cos_med <-0.15 y r<10 = direccion (PCGrad sobre {pose+root, foot}).
  |cos|<0.05 y r<3 = ortogonales (revisar mascara/seeds, no pesos).
Con ~32k params el coseno aleatorio es ~0.006: |cos|<0.05 = ortogonal.

Uso: .venv/bin/python tools/grad_audit.py --ckpt /tmp/opencode/ckpts/v5_film1_best.pt --n-batches 40
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from train import build_windows, unroll  # noqa: E402
from mono_model import MonoNet, MonoFiLM  # noqa: E402
from preprocess import parse_bvh  # noqa: E402


def trunk_groups(model):
    if isinstance(model, MonoFiLM):
        return {"W1": list(model.fc1.parameters()), "W2": list(model.fc2.parameters()),
                "W3": list(model.fc3.parameters()), "film": list(model.film.parameters()),
                "all": list(model.fc1.parameters()) + list(model.fc2.parameters())}
    if isinstance(model, MonoNet):
        return {"W1": list(model.net[0].parameters()), "W2": list(model.net[2].parameters()),
                "W3": list(model.net[4].parameters()),
                "all": list(model.net[0].parameters()) + list(model.net[2].parameters())}
    return {"all": list(model.parameters())}


def flat(gs, ps):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                      for g, p in zip(gs, ps)])


def main():
    a = parse_args()
    ckpt = torch.load(a.ckpt, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    arch = "moe" if any("router" in k for k in state.keys()) else "mono"
    assert arch == "mono", "auditoria solo para tronco mono/film"
    model = MonoFiLM() if any("film" in k for k in state.keys()) else MonoNet()
    model.load_state_dict(state)
    model.train()  # gradientes en modo train, como en el loop
    W = {"pose": 1.0, "root": float(cfg.get("w_root", 1.0)),
         "foot": float(cfg.get("w_foot", 0.25)), "contact": float(cfg.get("w_contact", 0.25)),
         "limit": float(cfg.get("w_limit", 0.15))}
    print(f"pesos efectivos: {W}")
    bundle = np.load(SRC.parent / "data" / "locomotion.npz")
    inputs = torch.from_numpy(bundle["inputs"])
    targets = torch.from_numpy(bundle["targets"])
    train_end = int(bundle["train_end"])
    frame_time = float(bundle["frame_time"])
    phase_period = float(bundle["phase_period"])
    root_mean = torch.from_numpy(bundle["root_mean"].astype(np.float32))
    root_std = torch.from_numpy(bundle["root_std"].astype(np.float32))
    vel_mean = torch.from_numpy(bundle["vel_mean"].astype(np.float32))
    vel_std = torch.from_numpy(bundle["vel_std"].astype(np.float32))
    feet_mean = torch.from_numpy(bundle["feet_mean"].astype(np.float32))
    feet_std = torch.from_numpy(bundle["feet_std"].astype(np.float32))
    euler_min = torch.from_numpy(bundle["euler_min"])
    euler_max = torch.from_numpy(bundle["euler_max"])
    names, parents, offsets, _, _, _ = parse_bvh(
        SRC.parent / "dataset" / "additional_basic_locomotion_2.bvh")
    off = torch.tensor(np.array(offsets), dtype=torch.float32)
    inf_idx = sorted(set(int(j) for j in bundle["inferior_joints"]))
    sup_idx = sorted(set(int(j) for j in bundle["superior_joints"]))
    starts = torch.arange(1, train_end - 8 + 1)
    loader = DataLoader(TensorDataset(starts), batch_size=a.batch, shuffle=True)
    groups = trunk_groups(model)
    rows = {g: [] for g in groups}
    nb, nskip = 0, 0
    for (sb,) in loader:
        if nb >= a.n_batches:
            break
        true_win, prev_win = build_windows(targets, sb, 8)
        true_root_win = true_win[:, :, 92:95] * root_std + root_mean
        root_prev = root_mean + targets[sb - 1][:, 92:95] * root_std
        out = unroll(model, inputs[sb], root_prev, true_win, prev_win, true_root_win,
                     off, parents, inf_idx, sup_idx, root_mean, root_std,
                     vel_mean, vel_std, feet_mean, feet_std, euler_min, euler_max,
                     frame_time, phase_period, w_vel=0.2, w_contact=W["contact"],
                     w_foot=W["foot"], w_limit=W["limit"], p_pred=0.5,
                     detach_carry=False, arch="mono")
        if out["contact"].item() == 0:
            nskip += 1
            continue
        L = {"pose": out["pose"], "root": out["root"], "foot": out["foot"],
             "contact": out["contact"], "limit": out["limit"]}
        for gn, ps in groups.items():
            g = {k: flat(torch.autograd.grad(W[k] * L[k], ps, retain_graph=True,
                                             allow_unused=True), ps) for k in L}
            # El sliding batch-dim MUERE con shuffle (0.0 exacto): el termino
            # vivo de pies es contact (mascara GT). push = pose+root.
            push = g["pose"] + g["root"]
            for target in ("contact", "foot"):
                gt = g[target]
                fn = gt.norm() + 1e-12
                if gt.norm().item() == 0:
                    rows[gn].append((target, float("inf"), 0.0, 0.0, 0.0, 0.0))
                    continue
                rows[gn].append((target, float((push.norm() / fn).item()),
                                 float(F.cosine_similarity(gt, push, dim=0).item()),
                                 float((F.cosine_similarity(gt, push, dim=0).item()) < 0),
                                 float(F.cosine_similarity(gt, g["pose"], dim=0).item()),
                                 float(F.cosine_similarity(gt, g["root"], dim=0).item())))
        nb += 1
    print(f"batches {nb} (sin apoyo: {nskip})")
    for gn, rs in rows.items():
        for target in ("contact", "foot"):
            arr = np.array([r[1:] for r in rs if r[0] == target])
            if len(arr) == 0:
                continue
            if np.isinf(arr[:, 0]).all():
                print(f"[{gn}/{target}] MUERTO (grad 0 en batch shuffled -> --foot-mode time)")
                continue
            arr = arr[~np.isinf(arr[:, 0])]
            r_med, r_p90 = float(np.median(arr[:, 0])), float(np.quantile(arr[:, 0], 0.9))
            cos_med = float(np.median(arr[:, 1]))
            frac_neg = float(np.mean(arr[:, 2]))
            print(f"[{gn}/{target}] r_med {r_med:.2f} r_p90 {r_p90:.2f} cos_med {cos_med:+.3f} "
                  f"frac_neg {frac_neg:.2f} cos_pose {np.median(arr[:, 3]):+.3f} "
                  f"cos_root {np.median(arr[:, 4]):+.3f}")
            if r_med > 10:
                print("  -> MAGNITUD: ahogado (apuesta 5 o w x r/3; PCGrad no)")
            elif frac_neg > 0.4 and cos_med < -0.15 and r_med < 10:
                print("  -> DIRECCION: compiten (PCGrad; si persiste, cabeza propia)")
            elif abs(cos_med) < 0.05 and r_med < 3:
                print("  -> ORTOGONAL: el loss no mueve la metrica (mascara/seeds)")


def parse_args():
    p = argparse.ArgumentParser(description="Auditoria de gradientes por termino")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-batches", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    main()
