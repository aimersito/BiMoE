#!/usr/bin/env python3
"""Entrenamiento BiMoE-10K v5: velocity lineal + contact + fase + limite 0.15.

v5 ataca los 3 numeros rojos medidos con v4-ep20 (pies 0.28m, limite 0.115,
error x7 en 24f) con trucos de la literatura, sin reinventar:
1. VELOCITY LINEAL MSE(delta_pred, delta_gt) 0.2 (siMLPe/Back-to-MLP):
   reemplaza al log1p (sin evidencia para velocidad de joints; amplifica ruido).
2. CONTACT 0.5 con auto-label GT (altura<0.10m AND vel<0.15m/s, Kimodo) sobre
   pies predichos (HuMoR/MDM). Ataca pies flotantes directo.
3. FASE como input (1 float, truco PFNN barato): avance determinista.
4. LIMITE 0.5->0.15: a 16k params el auxiliar ahogaba a pose (era 29% del total).
Se conserva: residual, sampling 0->0.5, root supervisado, BPTT truncado, clamps.

Uso:  python3 src/train.py --data data/locomotion.npz --epochs 200 --unroll 8 --out weights/
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader, TensorDataset

from bimoe_model import BiMoE
from mono_model import MonoNet, predict_next
from losses import (assemble_prediction, foot_contact_loss, foot_sliding_loss,
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


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val, config):
    """Guarda checkpoint completo: pesos + optimizador + scheduler + config.
    Sin esto no hay resume real (el optimizador fresco rompe la continuidad)."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val": best_val,
        "config": config,
    }, path)


def load_resume(path, model, optimizer, scheduler):
    """Carga checkpoint nuevo (dict) o legacy (solo pesos, optimizador fresco)."""
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"  [resume] scheduler no cargado ({e}), se sigue igual")
        print(f"  [resume] {path} -> epoch {ckpt.get('epoch')} best {ckpt.get('best_val')}")
        return int(ckpt.get("epoch", 0)), float(ckpt.get("best_val", float("inf")))
    model.load_state_dict(ckpt)
    print(f"  [resume] {path} -> pesos legacy (optimizador fresco, una sola vez)")
    return 0, float("inf")


def rollout_120f_metric(model, arch, tq_t, tr_np, phase_np, period, seeds,
                        stats_d, offsets_d, parents, inf_idx, sup_idx, dt):
    """Rollout game-like 120f liviano para el log: mediana de pies (m) en seeds.
    Misma receta que eval_gates (historial propio, root con velocidad real,
    fase determinista). Sin gradiente."""
    rm, rs = stats_d["root_mean"], stats_d["root_std"]
    vm, vs = stats_d["vel_mean"], stats_d["vel_std"]
    fm, fs = stats_d["feet_mean"], stats_d["feet_std"]
    true_vel = (tr_np[1:] - tr_np[:-1]) / dt
    meds = []
    with torch.no_grad():
        for s in seeds:
            hist = [tq_t[s].clone(), tq_t[s - 1].clone(), tq_t[s - 2].clone()]
            root = tr_np[s].copy()
            ph = float(phase_np[s + 3] % 1.0)
            qs, rs_ = [], []
            for _ in range(120):
                vv = true_vel[min(s + len(qs), len(true_vel) - 1)]
                root = root + vv * dt
                root_t = torch.from_numpy(root).to(device=tq_t.device, dtype=torch.float32)
                vel_t = torch.from_numpy(vv).to(device=tq_t.device, dtype=torch.float32)
                pos = forward_kinematics_batch(hist[0].reshape(1, 23, 4),
                                               root_t.reshape(1, 3), offsets_d, parents)
                feet = torch.stack([torch.minimum(pos[0, 21], pos[0, 22]),
                                    torch.minimum(pos[0, 17], pos[0, 18])])[:, 1]
                state = torch.cat([x.reshape(-1) for x in hist] +
                                  [(root_t - rm) / rs, (vel_t - vm) / vs,
                                   (feet - fm) / fs,
                                   torch.tensor([ph], device=tq_t.device)])
                if arch == "mono":
                    nxt, _ = predict_next(model(state.unsqueeze(0)),
                                          hist[0].reshape(1, 23, 4),
                                          root_t.reshape(1, 3), rs)
                    nxt = nxt.reshape(23, 4)
                else:
                    lower, upper, _ = model(state)
                    dd = torch.zeros(23, 4, device=tq_t.device)
                    dd[inf_idx] = lower[:36].reshape(-1, 4)
                    dd[sup_idx] = upper.reshape(-1, 4)
                    nxt = hist[0] + dd
                    nxt = nxt / nxt.norm(dim=-1, keepdim=True).clamp_min(1e-4)
                hist = [nxt, hist[0], hist[1]]
                ph = (ph + 1.0 / period) % 1.0
                qs.append(nxt)
                rs_.append(root.copy())
            q = torch.stack(qs)
            r = torch.from_numpy(np.array(rs_)).to(device=tq_t.device, dtype=torch.float32)
            joints = forward_kinematics_batch(q, r, offsets_d, parents)
            feet_h = torch.minimum(torch.minimum(joints[:, 21, 1], joints[:, 22, 1]),
                                   torch.minimum(joints[:, 17, 1], joints[:, 18, 1]))
            meds.append(float(feet_h.median().item()))
    return float(sum(meds) / len(meds))


def unroll(model, state0, root_prev_raw, true_win, prev_true_win, true_root_win, offsets_t, parents,
           inferior_idx, superior_idx, root_mean, root_std, vel_mean, vel_std,
           feet_mean, feet_std, euler_min, euler_max, frame_time, phase_period,
           w_vel=0.2, w_contact=0.5, w_foot=0.5, w_limit=0.15,
           p_pred=1.0, detach_carry=True, arch="moe"):
    """Desenrolla K pasos. v5 = v4 + velocity lineal + contact + fase + limite 0.15.
    - VELOCITY LINEAL MSE(delta_pred, delta_gt) 0.2: reemplaza al log1p (el log1p
      amplifica ruido en trote; sin evidencia publicada para velocidad de joints).
    - CONTACT 0.5: etiquetas del GT (altura<0.10m AND vel<0.15m/s, Kimodo) sobre
      pies predichos (HuMoR/MDM). Ataca pies flotantes directo.
    - FASE: input 285, avance determinista fase += 1/periodo (truco PFNN barato).
    - LIMITE 0.15: a 16k params el 0.5 ahogaba a pose (29% del total).
    p_pred = prob. de realimentar la prediccion propia (rampa por epoca).
    detach_carry=True = BPTT truncado (estable, bendecido por pushforward trick).
    true_root_win: (B,K,3) raices reales en metros (para sampling + loss root)."""
    n_joints = 23
    state = state0
    root_prev = root_prev_raw
    pose_sum = 0.0
    root_sum = 0.0
    foot_sum = 0.0
    limit_sum = 0.0
    vel_sum = 0.0
    stacked = []
    pred_feet = []
    steps = true_win.shape[1]
    # Etiquetas de contacto desde el GT (sin gradiente): una sola FK del window.
    with torch.no_grad():
        true_pos = forward_kinematics_batch(true_win[:, :, :92].reshape(-1, steps, n_joints, 4).reshape(-1, n_joints, 4),
                                            true_root_win.reshape(-1, 3), offsets_t, parents)
        true_pos = true_pos.reshape(-1, steps, n_joints, 3)
        true_feet = torch.stack([torch.minimum(true_pos[:, :, 21], true_pos[:, :, 22]),
                                 torch.minimum(true_pos[:, :, 17], true_pos[:, :, 18])], dim=2)
        true_speed = torch.diff(true_feet[:, :, :, [0, 2]], dim=1).norm(dim=-1) / frame_time
        contact_mask = (true_feet[:, 1:, :, 1] < 0.10) & (true_speed < 0.15)  # (B,K-1,2)
    for k in range(steps):
        h0 = state[:, :92].reshape(-1, n_joints, 4)
        if arch == "mono":
            full, root_raw = predict_next(model(state), h0, root_prev, root_std)
        else:
            lower, upper, _ = model(state)
            full, root_raw, _ = assemble_prediction(lower, upper, h0, root_prev,
                                                    inferior_idx, superior_idx,
                                                    root_mean, root_std, n_joints)
        true_full = true_win[:, k, :92].reshape(-1, n_joints, 4)
        true_root = true_root_win[:, k]
        pose_sum = pose_sum + quaternion_sign_invariant_loss(full, true_full)
        root_sum = root_sum + (((root_raw - true_root) / root_std).pow(2)).mean()
        foot_sum = foot_sum + foot_sliding_loss(full, root_raw, offsets_t, parents,
                                                     (17, 21), (18, 22))
        limit_sum = limit_sum + joint_limit_loss(full, euler_min, euler_max)
        # Velocity LINEAL sobre deltas (v5): MSE directo, sin log1p ni denominador.
        true_delta = true_full - prev_true_win[:, k]
        vel_sum = vel_sum + ((full - h0) - true_delta).pow(2).mean()
        stacked.append(full)
        # Pies predichos para contact (con gradiente; FK propia del paso).
        pos_pred = forward_kinematics_batch(full, root_raw, offsets_t, parents)
        pred_feet.append(torch.stack([torch.minimum(pos_pred[:, 21], pos_pred[:, 22]),
                                      torch.minimum(pos_pred[:, 17], pos_pred[:, 18])], dim=1))
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
        # Fase determinista (no se predice): avanza 1/periodo por frame.
        phase_next = (state[:, -1:] + 1.0 / phase_period) % 1.0
        state = torch.cat([carry_q.reshape(batch, -1),
                           state[:, :184], root_n, vel_n, feet_n, phase_next], dim=1)
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
    # Contact sobre pies predichos apilados (con gradiente) + etiquetas GT.
    pred_feet_seq = torch.stack(pred_feet, dim=1)  # (B, K, 2, 3)
    if steps >= 2:
        contact = foot_contact_loss(pred_feet_seq, contact_mask)
    else:
        contact = pred_feet_seq.sum() * 0.0
    n = float(steps)
    return {"pose": pose_sum / n, "root": root_sum / n, "foot": foot_sum / n,
            "limit": limit_sum / n, "vel": vel_sum / n, "contact": contact,
            "smooth": smooth, "stacked": stacked}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device} | unroll: {args.unroll} | vel: {args.w_vel} "
          f"contact: {args.w_contact} limit: {args.w_limit}")

    bundle = np.load(args.data)
    inputs = torch.from_numpy(bundle["inputs"])
    targets = torch.from_numpy(bundle["targets"])
    train_end, val_end = int(bundle["train_end"]), int(bundle["val_end"])
    frame_time = float(bundle["frame_time"])
    phase_period = float(bundle["phase_period"])
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
    # Datos + stats chicos (~8 MB): a GPU una sola vez, cero transferencias por batch.
    # (Todo junto aqui: inputs/targets se usan con las stats en la misma expresion.)
    if device == "cuda":
        inputs, targets = inputs.cuda(), targets.cuda()
        root_mean, root_std = root_mean.cuda(), root_std.cuda()
        vel_mean, vel_std = vel_mean.cuda(), vel_std.cuda()
        feet_mean, feet_std = feet_mean.cuda(), feet_std.cuda()
        euler_min, euler_max = euler_min.cuda(), euler_max.cuda()

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

    model = (MonoNet() if args.arch == "mono" else BiMoE()).to(device)
    prefix = "mono1x" if args.arch == "mono" else "bimoe10k"
    run_id = args.run_id or prefix
    print(f"Arquitectura: {args.arch} | Parametros: {model.count_parameters()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Metodologia un solo modelo: sin --resume ni --fresh y con checkpoints
    # previos del mismo run_id -> abortar (preguntar antes de reiniciar).
    if not args.resume and not args.fresh:
        prev = sorted(out_dir.glob(f"{run_id}_*.pt"))
        if prev:
            print(f"ABORTO: hay {len(prev)} checkpoint(s) de '{run_id}' en {out_dir}:")
            for p in prev[-5:]:
                print(f"  - {p.name}")
            print("Usa --resume <ckpt> para continuar o --fresh para reiniciar "
                  "(solo si cambias de arquitectura).")
            raise SystemExit(1)
    config = {"arch": args.arch, "run_id": run_id, "unroll": args.unroll,
              "batch": args.batch, "noise": args.noise,
              "w_vel": args.w_vel, "w_contact": args.w_contact,
              "w_foot": args.w_foot, "w_limit": args.w_limit,
              "ss_end": args.ss_end, "ss_ramp": args.ss_ramp,
              "full_bptt": args.full_bptt, "val_every": args.val_every,
              "min_epochs": args.min_epochs, "patience": args.patience}
    start_epoch, best_val = 0, float("inf")
    if args.resume:
        start_epoch, best_val = load_resume(args.resume, model, optimizer, scheduler)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}_{datetime.date.today():%Y%m%d}.log"
    with open(log_path, "a") as f:
        f.write(f"# run {run_id} arch={args.arch} resume={args.resume or '-'} "
                f"start_epoch={start_epoch} config={json.dumps(config)}\n")
    print(f"Checkpoints -> {out_dir} | log -> {log_path} | resume desde epoch {start_epoch}")
    detach_carry = not args.full_bptt
    print(f"BPTT: {'completo (encadenado)' if args.full_bptt else 'truncado (estable)'}")
    patience_left = args.patience
    val_every = max(1, args.val_every)
    # Cache del ultimo val (las epochs sin val loguean train + ultimo val).
    last = {"val": float("nan"), "pose": float("nan"), "root": float("nan"),
            "vel": float("nan"), "contact": float("nan"), "foot": float("nan"),
            "limit": float("nan"), "ref": float("nan"), "roll": float("nan")}
    dev = torch.device(device)
    off_d = offsets_t.to(dev)
    # Tensores para el rollout 120f del log (game-like, velocidad real).
    tq_all = torch.from_numpy(bundle["targets"][:, :92]).reshape(-1, 23, 4).float().to(dev)
    tr_all = (torch.from_numpy(bundle["targets"][:, 92:95].astype(np.float32)).to(dev)
              * root_std.to(dev) + root_mean.to(dev)).double().cpu().numpy()
    phase_all = bundle["phase"].astype(np.float64)
    stats_d = {"root_mean": root_mean.to(dev), "root_std": root_std.to(dev),
               "vel_mean": vel_mean.to(dev), "vel_std": vel_std.to(dev),
               "feet_mean": feet_mean.to(dev), "feet_std": feet_std.to(dev)}
    n_targets = len(tq_all)
    roll_seeds = [min(val_end + 10 + i * 80, n_targets - 130)
                  for i in range(max(1, args.rollout_seeds))]

    for epoch in range(start_epoch + 1, args.epochs + 1):
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
                         euler_min.to(dev), euler_max.to(dev), frame_time, phase_period,
                         w_vel=args.w_vel, w_contact=args.w_contact,
                         w_foot=args.w_foot, w_limit=args.w_limit,
                         p_pred=p_pred, detach_carry=detach_carry, arch=args.arch)
            reg = sum(p.pow(2).sum() for p in model.parameters())
            loss = (out["pose"] + out["root"] + args.w_foot * out["foot"]
                    + 0.3 * out["smooth"] + args.w_limit * out["limit"]
                    + args.w_vel * out["vel"] + args.w_contact * out["contact"]
                    + 1e-5 * reg)
            if not torch.isfinite(loss):
                with torch.no_grad():
                    diag = {k: (v.item() if torch.is_tensor(v) else v)
                            for k, v in {"pose": out["pose"], "root": out["root"],
                                         "foot": out["foot"], "contact": out["contact"],
                                         "smooth": out["smooth"], "limit": out["limit"],
                                         "vel": out["vel"]}.items()}
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

        # Val con cadencia (el grueso del costo por epoch): ep1, cada val_every y ultima.
        do_val = (epoch == 1) or (epoch % val_every == 0) or (epoch == args.epochs)
        val_finite = False
        if do_val:
            model.eval()
            with torch.no_grad():
                # Val con el MISMO p_pred final (mide el regimen del juego).
                v = unroll(model, val_state0.to(dev), val_root_prev.to(dev),
                           val_true.to(dev), val_prev.to(dev), val_true_root.to(dev),
                           off_d, parents,
                           inferior_idx, superior_idx, root_mean.to(dev), root_std.to(dev),
                           vel_mean.to(dev), vel_std.to(dev), feet_mean.to(dev), feet_std.to(dev),
                           euler_min.to(dev), euler_max.to(dev), frame_time, phase_period,
                           w_vel=args.w_vel, w_contact=args.w_contact,
                           w_foot=args.w_foot, w_limit=args.w_limit,
                           p_pred=args.ss_end, arch=args.arch)
                val_tensor = (v["pose"] + v["root"] + args.w_foot * v["foot"]
                              + args.w_limit * v["limit"] + args.w_vel * v["vel"]
                              + args.w_contact * v["contact"])
                val_finite = bool(torch.isfinite(val_tensor).item())
                val_loss = val_tensor.item() if val_finite else float("nan")
                # Referencia one-step clasica sobre pares de val (base real -> target).
                if args.arch == "mono":
                    d95 = model(val_state0[:512].to(dev))
                    bq = val_state0[:512, :92].reshape(-1, 23, 4).to(dev)
                    br = (val_state0[:512, 276:279].to(dev) * root_std.to(dev)
                          + root_mean.to(dev))
                    fq, _ = predict_next(d95, bq, br, root_std.to(dev))
                    tq0 = val_true[:512, 0, :92].reshape(-1, 23, 4).to(dev)
                    ref = quaternion_sign_invariant_loss(fq, tq0)
                else:
                    lo, up, _ = model(val_state0[:512].to(dev))
                    ref, _ = total_loss(lo, up, val_state0[:512].to(dev), val_true[:512, 0].to(dev),
                                        off_d, parents,
                                        inferior_idx, superior_idx, root_mean.to(dev),
                                        root_std.to(dev), euler_min.to(dev), euler_max.to(dev))
                if val_finite:
                    last.update({"val": val_loss, "pose": v["pose"].item(),
                                 "root": v["root"].item(), "vel": v["vel"].item(),
                                 "contact": v["contact"].item(), "foot": v["foot"].item(),
                                 "limit": v["limit"].item(), "ref": ref.item()})
            # Rollout 120f liviano para el log (solo en epochs de val).
            if epoch % max(1, args.rollout_every) == 0:
                with torch.no_grad():
                    last["roll"] = rollout_120f_metric(model, args.arch, tq_all, tr_all,
                                                       phase_all, phase_period, roll_seeds,
                                                       stats_d, off_d, parents,
                                                       inferior_idx, superior_idx,
                                                       frame_time)
        tag = "" if do_val else " [sin val]"
        line = (f"epoch {epoch:3d} train {train_loss / len(train_loader):.6f} "
                f"val_unroll {last['val']:.6f} (pose {last['pose']:.6f} "
                f"root {last['root']:.6f} vel {last['vel']:.6f} "
                f"contact {last['contact']:.6f} foot {last['foot']:.6f} "
                f"limit {last['limit']:.6f}) ref1 {last['ref']:.6f} "
                f"roll120_feet {last['roll']:.3f} p {p_pred:.2f}{tag}")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

        if do_val:
            if val_finite and val_loss < best_val:
                best_val = val_loss
                patience_left = args.patience
                save_checkpoint(out_dir / f"{run_id}_best.pt", epoch, model, optimizer,
                                scheduler, best_val, config)
            else:
                # Cada chequeo fallido descuenta val_every epochs (regla: 10 epochs).
                patience_left -= val_every
                if patience_left <= 0 and epoch >= args.min_epochs:
                    print("Early stopping.")
                    break
        if epoch % max(1, args.ckpt_every) == 0:
            save_checkpoint(out_dir / f"{run_id}_epoch{epoch}_pose{last['pose']:.4f}_foot{last['foot']:.4f}.pt",
                            epoch, model, optimizer, scheduler, best_val, config)
    print(f"Mejor val_unroll: {best_val:.6f} -> {out_dir / f'{run_id}_best.pt'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena BiMoE-10K v3 (desenrollado)")
    parser.add_argument("--arch", default="moe", choices=["moe", "mono"],
                        help="moe = router+2 expertos (BiMoE), mono = una sola red (1X)")
    parser.add_argument("--data", default="data/locomotion.npz")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.01,
                        help="ruido gaussiano en quats de entrada (robustez)")
    parser.add_argument("--unroll", type=int, default=8,
                        help="pasos autoalimentados por batch (anti-congelamiento)")
    parser.add_argument("--full-bptt", action="store_true",
                        help="gradiente encadenado entre pasos (por defecto: truncado)")
    parser.add_argument("--w_vel", type=float, default=0.2,
                        help="peso velocity lineal MSE(deltas) (v5; reemplaza log1p)")
    parser.add_argument("--w_contact", type=float, default=0.5,
                        help="peso contact con etiquetas GT altura+vel (v5)")
    parser.add_argument("--w_foot", type=float, default=0.5,
                        help="peso foot sliding")
    parser.add_argument("--w_limit", type=float, default=0.15,
                        help="peso limite anatomico (v5: 0.5 ahogaba a pose)")
    parser.add_argument("--ss-end", type=float, default=0.5,
                        help="scheduled sampling final: prob. de realimentar prediccion")
    parser.add_argument("--ss-ramp", type=int, default=60,
                        help="epocas para subir el sampling de 0 a ss-end")
    parser.add_argument("--out", default="weights")
    parser.add_argument("--resume", default=None,
                        help="checkpoint para continuar (pesos + optimizador)")
    parser.add_argument("--fresh", action="store_true",
                        help="reiniciar desde cero aunque haya checkpoints (solo cambio de arquitectura)")
    parser.add_argument("--run-id", default=None,
                        help="id del run para checkpoints/logs (default: mono1x/bimoe10k)")
    parser.add_argument("--ckpt-every", type=int, default=5,
                        help="cada cuantas epochs guarda checkpoint descriptivo")
    parser.add_argument("--patience", type=int, default=10,
                        help="early stopping: epochs sin mejorar val_unroll (descuenta val-every por chequeo)")
    parser.add_argument("--min-epochs", type=int, default=40,
                        help="el early stopping no actua antes de esta epoch (evita matar runs bebes)")
    parser.add_argument("--rollout-every", type=int, default=5,
                        help="cada cuantas epochs loguea rollout 120f (solo log, no toca entreno)")
    parser.add_argument("--rollout-seeds", type=int, default=2,
                        help="seeds del rollout 120f del log (liviano)")
    parser.add_argument("--val-every", type=int, default=5,
                        help="cada cuantas epochs corre val+ref+rollout (el grueso del costo por epoch)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
