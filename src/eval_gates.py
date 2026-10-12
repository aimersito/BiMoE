#!/usr/bin/env python3
"""Puertas v4: se corren en Kaggle ANTES de cantar victoria (segundos en GPU).

Puerta 1 (sanidad Martinez): rollout corto autoalimentado vs baseline
  zero-velocity a 5/10/20/24 frames. El modelo debe GANAR en todos.
Puerta 2 (juego): rollout 120 frames con trayectoria de velocidad REAL:
  pies medianos < 0.08 m, jerk > 20% del natural (no congelado),
  violaciones de limite ~0.
Todo con salidas RESIDUALES (v4): next = base + delta.

Uso:  python3 src/eval_gates.py --weights weights/bimoe10k_best.pt [--mode all]
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
from mono_model import MonoNet, MonoFiLM, RootNet, predict_next
from losses import forward_kinematics_batch, joint_limit_loss
from preprocess import parse_bvh


def load_all(weights_path):
    bundle = np.load(ROOT / "data" / "locomotion.npz")
    names, parents, offsets, _, _, frame_time = parse_bvh(
        ROOT / "dataset" / "additional_basic_locomotion_2.bvh")
    ckpt = torch.load(weights_path, map_location="cpu")
    # Formato nuevo (dict con model_state_dict) o legacy (solo pesos).
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print(f"Checkpoint: epoch {ckpt.get('epoch')} best {ckpt.get('best_val')}")
    # Autodeteccion: keys con 'router' = MoE, con 'film' = mono-FiLM, si no = mono (1X).
    arch = "moe" if any("router" in k for k in state.keys()) else "mono"
    if arch == "mono" and any("film" in k for k in state.keys()):
        model = MonoFiLM()
        print("Variante detectada: FiLM (fase modula ocultas)")
    else:
        model = BiMoE() if arch == "moe" else MonoNet()
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
    # Stats de reparam (apuesta 3): mismos datos -> mismos valores que en train.
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
    stats = {k: torch.from_numpy(bundle[k].astype(np.float32))
             for k in ["root_mean", "root_std", "vel_mean", "vel_std",
                       "feet_mean", "feet_std", "euler_min", "euler_max"]}
    tq = torch.from_numpy(bundle["targets"][:, :92]).reshape(-1, 23, 4).float()
    tr = (torch.from_numpy(bundle["targets"][:, 92:95].astype(np.float32))
          * stats["root_std"] + stats["root_mean"]).numpy().astype(np.float64)
    # Fase v5: targets[i] = frame i+3 del BVH.
    phase = bundle["phase"].astype(np.float64)
    period = float(bundle["phase_period"])
    return model, bundle, parents, torch.tensor(np.array(offsets), dtype=torch.float32), \
        float(frame_time), stats, tq, tr, phase, period, arch, root_net, reparam, dq_std, delta_std


def reparam_state_eval(state, dq_std):
    q0, q1, q2 = state[:, :92], state[:, 92:184], state[:, 184:276]
    return torch.cat([q0, (q0 - q1) / dq_std, (q1 - q2) / dq_std, state[:, 276:]], dim=1)


def rollout(model, arch, tq, tr, phase, period, seed_idx, steps, stats, offsets_t, parents, dt,
            inf_idx, sup_idx, true_vel_traj=True, root_net=None, reparam=False,
            dq_std=None, delta_std=None):
    """Rollout game-like: historial propio, root integrado con velocidad REAL.
    v5: fase determinista (fase += 1/periodo por frame)."""
    hist = [tq[seed_idx].clone(), tq[seed_idx - 1].clone(), tq[seed_idx - 2].clone()]
    root = tr[seed_idx].copy()
    ph = float(phase[seed_idx + 3] % 1.0)
    true_vel = (tr[1:] - tr[:-1]) / dt
    quats_out, roots_out = [], []
    rm, rs, vm, vs, fm, fs = (stats["root_mean"], stats["root_std"], stats["vel_mean"],
                              stats["vel_std"], stats["feet_mean"], stats["feet_std"])
    with torch.no_grad():
        for h in range(steps):
            v = true_vel[min(seed_idx + h, len(true_vel) - 1)]
            root = root + v * dt
            root_t = torch.from_numpy(root).float()
            vel_t = torch.from_numpy(v).float()
            pos = forward_kinematics_batch(hist[0].reshape(1, 23, 4).float(),
                                           root_t.reshape(1, 3), offsets_t, parents)
            feet = torch.stack([torch.minimum(pos[0, 21], pos[0, 22]),
                                torch.minimum(pos[0, 17], pos[0, 18])])[:, 1]
            state = torch.cat([x.reshape(-1) for x in hist] +
                              [(root_t - rm) / rs, (vel_t - vm) / vs, (feet - fm) / fs,
                               torch.tensor([ph])])
            if reparam:
                state = reparam_state_eval(state.unsqueeze(0), dq_std).squeeze(0)
            if arch == "mono":
                d95 = model(state.unsqueeze(0))
                if delta_std is not None:
                    d95 = torch.cat([d95[:, :92] * delta_std, d95[:, 92:95]], dim=1)
                if root_net is not None:
                    d95 = torch.cat([d95[:, :92], root_net(state.unsqueeze(0))], dim=1)
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
                nxt = nxt / nxt.norm(dim=-1, keepdim=True)
            hist = [nxt, hist[0], hist[1]]
            ph = (ph + 1.0 / period) % 1.0
            quats_out.append(nxt.numpy())
            roots_out.append(root.copy())
    return np.array(quats_out), np.array(roots_out)


def quat_mse(a, b):
    return np.minimum(((a - b) ** 2).sum(axis=-1), ((a + b) ** 2).sum(axis=-1)).mean()


def cv_baseline(tq, seed_idx, steps):
    """Extrapolacion a velocidad constante: q + h*dq, renormalizado. M3."""
    base = tq[seed_idx]
    dq = tq[seed_idx] - tq[seed_idx - 1]
    out = []
    for h in range(1, steps + 1):
        q = base + h * dq
        out.append((q / q.norm(dim=-1, keepdim=True).clamp_min(1e-4)).numpy())
    return np.array(out)


def gate_short(model, arch, tq, tr, phase, period, seeds, stats, offsets_t, parents, dt, inf_idx, sup_idx,
               horizons=(5, 10, 20, 24), split_out=False, cv=False, root_net=None,
               reparam=False, dq_std=None, delta_std=None):
    print("== Puerta 1: modelo vs zero-velocity (menor es mejor) ==")
    ok = True
    for h in horizons:
        m_errs, z_errs, c_errs = [], [], []
        for s in seeds:
            pred, _ = rollout(model, arch, tq, tr, phase, period, s, h, stats, offsets_t, parents,
                              dt, inf_idx, sup_idx, root_net=root_net, reparam=reparam,
                              dq_std=dq_std, delta_std=delta_std)
            real = tq[s + 1:s + 1 + h].numpy()
            m_errs.append(quat_mse(pred, real[:len(pred)]))
            zf = np.repeat(tq[s].numpy()[None], len(pred), axis=0)
            z_errs.append(quat_mse(zf, real[:len(pred)]))
            if cv:
                c_errs.append(quat_mse(cv_baseline(tq, s, h), real[:h]))
        m, z = float(np.mean(m_errs)), float(np.mean(z_errs))
        win = m < z
        ok = ok and win
        extra = ""
        if split_out:
            extra += f" ratio {m / max(z, 1e-12):.2f}x"
        if cv:
            extra += f" cv {np.mean(c_errs):.5f}"
        print(f"  {h:3d} frames ({h*dt*1000:4.0f} ms): modelo {m:.5f} vs zero-vel {z:.5f} "
              f"-> {'GANA' if win else 'PIERDE'}{extra}")
    print("  Puerta 1:", "PASS" if ok else "FAIL")
    return ok


def gate_rollout(model, arch, tq, tr, phase, period, seeds, stats, offsets_t, parents, dt, inf_idx, sup_idx,
                 root_net=None, reparam=False, dq_std=None, delta_std=None):
    print("== Puerta 2: rollout 120f game-like ==")
    feet_all, jerks, lims, frozen = [], [], [], 0
    nat_jerk = float(np.abs(np.diff(tq.numpy(), axis=0)).mean())
    for s in seeds:
        pred, roots = rollout(model, arch, tq, tr, phase, period, s, 120, stats, offsets_t, parents,
                              dt, inf_idx, sup_idx, root_net=root_net, reparam=reparam,
                              dq_std=dq_std, delta_std=delta_std)
        with torch.no_grad():
            pos = forward_kinematics_batch(torch.from_numpy(pred).float(),
                                           torch.from_numpy(roots).float(),
                                           offsets_t, parents).numpy()
        feet = np.minimum(np.minimum(pos[:, 21, 1], pos[:, 22, 1]),
                          np.minimum(pos[:, 17, 1], pos[:, 18, 1]))
        feet_all.append(np.median(feet))
        j = float(np.abs(np.diff(pred, axis=0)).mean())
        jerks.append(j)
        if j < 0.2 * nat_jerk:
            frozen += 1
        with torch.no_grad():
            lims.append(joint_limit_loss(torch.from_numpy(pred),
                                         stats["euler_min"],
                                         stats["euler_max"]).item())
    feet_med, jerk_med, lim_max = float(np.median(feet_all)), float(np.median(jerks)), max(lims)
    c1, c2, c3 = feet_med < 0.08, frozen == 0, lim_max < 0.01
    print(f"  pies mediana: {feet_med:.3f} m (<0.08: {'OK' if c1 else 'MAL'})")
    print(f"  jerk mediano: {jerk_med:.5f} (natural {nat_jerk:.5f}, congelados {frozen}/{len(seeds)})")
    print(f"  limite max: {lim_max:.5f} (<0.01: {'OK' if c3 else 'MAL'})")
    ok = c1 and c2 and c3
    print("  Puerta 2:", "PASS" if ok else "FAIL")
    return ok


def main():
    args = parse_args()
    model, bundle, parents, offsets_t, dt, stats, tq, tr, phase, period, arch, \
        root_net, reparam, dq_std, delta_std = load_all(args.weights)
    inf_idx = sorted(set(int(j) for j in bundle["inferior_joints"]))
    sup_idx = sorted(set(int(j) for j in bundle["superior_joints"]))
    va, n = int(bundle["val_end"]), len(tq)
    te = int(bundle["train_end"])
    horizons = tuple(int(h) for h in args.horizons.split(","))
    if args.windows == "train":
        seeds = sorted(set(min(10 + i * args.stride, te - 130) for i in range(args.n_seeds)))
        print(f"Ventanas TRAIN (M4): {len(seeds)} seeds stride {args.stride}")
    else:
        seeds = sorted(set(min(va + 10 + i * args.stride, n - 130) for i in range(args.n_seeds)))
    print(f"params: {model.count_parameters()} | seeds test: {seeds[0]}..{seeds[-1]} (n={len(seeds)} stride={args.stride})")
    ok1 = gate_short(model, arch, tq, tr, phase, period, seeds, stats, offsets_t, parents, dt,
                     inf_idx, sup_idx, horizons=horizons, split_out=args.split_out,
                     cv=args.cv, root_net=root_net, reparam=reparam,
                     dq_std=dq_std, delta_std=delta_std) if args.mode in ("all", "short") else True
    ok2 = gate_rollout(model, arch, tq, tr, phase, period, seeds, stats, offsets_t, parents, dt,
                       inf_idx, sup_idx, root_net=root_net, reparam=reparam,
                       dq_std=dq_std, delta_std=delta_std) if args.mode in ("all", "rollout") else True
    print("RESULTADO:", "PASS (a entrenar/demo)" if (ok1 and ok2) else "FAIL (iterar)")


def parse_args():
    p = argparse.ArgumentParser(description="Puertas v4")
    p.add_argument("--weights", default="weights/bimoe10k_best.pt")
    p.add_argument("--mode", default="all", choices=["all", "short", "rollout"])
    p.add_argument("--stride", type=int, default=80,
                   help="M7: stride de seeds (80 = regimen, 10 = 58 ventanas sin sesgo de fase)")
    p.add_argument("--n-seeds", type=int, default=8,
                   help="M7: numero de seeds (8 = regimen intacto)")
    p.add_argument("--horizons", default="5,10,20,24",
                   help="M1: horizontes P1 (default intacto; '1,2,3,5,10,20,24' = diagnostico)")
    p.add_argument("--split-out", action="store_true",
                   help="M1: reporta ratio modelo/zero por horizonte")
    p.add_argument("--cv", action="store_true",
                   help="M3: baseline velocidad-constante junto a modelo y zero-vel")
    p.add_argument("--windows", default="test", choices=["test", "train"],
                   help="M4: ventanas TEST (gates) o TRAIN (intercepto de ajuste vs generalizacion)")
    return p.parse_args()


if __name__ == "__main__":
    main()
