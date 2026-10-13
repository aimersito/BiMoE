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
from mono_model import MonoNet, MonoFiLM, RootNet, predict_next
from losses import (assemble_prediction, foot_contact_loss, foot_sliding_loss,
                    joint_limit_loss, quaternion_sign_invariant_loss,
                    quaternion_phuber_loss,
                    forward_kinematics_batch, total_loss)
from preprocess import parse_bvh


def build_windows(targets, starts, steps):
    """Ventanas de targets: true_win[k] y prev_true_win[k] (quats del frame previo)."""
    idx = starts[:, None] + torch.arange(steps)[None, :]
    true_win = targets[idx]                       # (B, K, 95)
    prev_idx = starts[:, None] - 1 + torch.arange(steps)[None, :]
    prev_true = targets[prev_idx][:, :, :92].reshape(-1, steps, 23, 4)
    return true_win, prev_true


def clip_valid_starts(clip_id, lo, hi, unroll, ahead=0):
    """Starts cuya ventana [s-1, s+unroll-1+ahead] vive dentro de un clip.

    Sin clip_id (npz v1) devuelve todo el rango (regimen intacto).
    """
    s = torch.arange(lo, hi)
    if clip_id is None:
        return s
    cid = clip_id if torch.is_tensor(clip_id) else torch.from_numpy(np.asarray(clip_id))
    far = (s + unroll - 1 + ahead).clamp(max=len(cid) - 1)
    return s[(cid[s - 1] == cid[far])]


def reparam_state(state, dq_std):
    """Reparam sin params (apuesta 3): [q0,q1,q2,rest] -> [q0,dq1,dq2,rest].
    dq1=(q0-q1)/std, dq2=(q1-q2)/std. Misma info, mejor condicionamiento:
    la velocidad deja de ser una resta O(1) que la red debe descubrir."""
    q0, q1, q2 = state[:, :92], state[:, 92:184], state[:, 184:276]
    rest = state[:, 276:]
    return torch.cat([q0, (q0 - q1) / dq_std, (q1 - q2) / dq_std, rest], dim=1)


def reparam_next_state(carry_q_flat, state, dq_std, batch):
    """Siguiente estado reparamado desde el carry (q0_new=carry, s1_new=(carry-s0)/std, s2_new=s1)."""
    s0 = state[:, :92]
    s1 = state[:, 92:184]
    return torch.cat([carry_q_flat, (carry_q_flat - s0) / dq_std, s1], dim=1)


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val, config,
                    ema_state=None, root_state=None):
    """Guarda checkpoint completo: pesos + optimizador + scheduler + config.
    Sin esto no hay resume real (el optimizador fresco rompe la continuidad).
    ema_state/root_state opcionales (apuestas 1-2, default off)."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val": best_val,
        "config": config,
        "ema_state_dict": ema_state,
        "root_state_dict": root_state,
    }, path)


def load_resume(path, model, optimizer, scheduler, root_net=None):
    """Carga checkpoint nuevo (dict) o legacy (solo pesos, optimizador fresco)."""
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"  [resume] optimizer fresco (grupo distinto: {e})")
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"  [resume] scheduler no cargado ({e}), se sigue igual")
        if root_net is not None:
            rs = ckpt.get("root_state_dict")
            if rs is not None:
                root_net.load_state_dict(rs)
                print("  [resume] root-net cargada")
            else:
                print("  [resume] root-net fresca (ckpt sin root)")
        print(f"  [resume] {path} -> epoch {ckpt.get('epoch')} best {ckpt.get('best_val')}")
        return int(ckpt.get("epoch", 0)), float(ckpt.get("best_val", float("inf"))), ckpt.get("ema_state_dict")
    model.load_state_dict(ckpt)
    print(f"  [resume] {path} -> pesos legacy (optimizador fresco, una sola vez)")
    return 0, float("inf"), None


def rollout_120f_metric(model, arch, tq_t, tr_np, phase_np, period, seeds,
                        stats_d, offsets_d, parents, inf_idx, sup_idx, dt,
                        root_net=None, reparam=False, dq_std=None, delta_std=None,
                        seed_lim=None, seed_per=None):
    """Rollout game-like 120f liviano para el log: mediana de pies (m) en seeds.
    Misma receta que eval_gates (historial propio, root con velocidad real,
    fase determinista). Sin gradiente. Soporta root-net/reparam/delta-scale.
    Multi-clip: seed_per[s] = periodo del clip propio; seed_lim[s] = ultimo
    indice valido (la velocidad real no cruza de clip). Sin ellos, regimen v1."""
    rm, rs = stats_d["root_mean"], stats_d["root_std"]
    vm, vs = stats_d["vel_mean"], stats_d["vel_std"]
    fm, fs = stats_d["feet_mean"], stats_d["feet_std"]
    true_vel = (tr_np[1:] - tr_np[:-1]) / dt
    meds = []
    with torch.no_grad():
        for s in seeds:
            per = (seed_per or {}).get(s, period)
            lim = (seed_lim or {}).get(s, len(true_vel) - 1)
            hist = [tq_t[s].clone(), tq_t[s - 1].clone(), tq_t[s - 2].clone()]
            root = tr_np[s].copy()
            ph = float(phase_np[s + 3] % 1.0)
            qs, rs_ = [], []
            for _ in range(120):
                vv = true_vel[min(s + len(qs), lim)]
                root = root + vv * dt
                root_t = torch.from_numpy(root).to(device=tq_t.device, dtype=torch.float32)
                vel_t = torch.from_numpy(vv).to(device=tq_t.device, dtype=torch.float32)
                pos = forward_kinematics_batch(hist[0].reshape(1, 23, 4),
                                               root_t.reshape(1, 3), offsets_d, parents)
                feet = torch.stack([torch.minimum(pos[0, 21], pos[0, 22]),
                                    torch.minimum(pos[0, 17], pos[0, 18])])[:, 1]
                state_raw = torch.cat([x.reshape(-1) for x in hist] +
                                  [(root_t - rm) / rs, (vel_t - vm) / vs,
                                   (feet - fm) / fs,
                                   torch.tensor([ph], device=tq_t.device)])
                state = (reparam_state(state_raw.unsqueeze(0), dq_std).squeeze(0)
                         if reparam else state_raw)
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
                    dd = torch.zeros(23, 4, device=tq_t.device)
                    dd[inf_idx] = lower[:36].reshape(-1, 4)
                    dd[sup_idx] = upper.reshape(-1, 4)
                    nxt = hist[0] + dd
                    nxt = nxt / nxt.norm(dim=-1, keepdim=True).clamp_min(1e-4)
                hist = [nxt, hist[0], hist[1]]
                ph = (ph + 1.0 / per) % 1.0
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
           w_vel=0.2, w_contact=0.5, w_foot=0.5, w_limit=0.15, w_dvel=0.0,
           p_pred=1.0, detach_carry=True, arch="moe",
           root_net=None, loss_shape="mse", phuber_eps=0.003,
           foot_shape="mse", w_step1=1.0, reparam=False, dq_std=None,
           delta_std=None, foot_mode="batch"):
    """Desenrolla K pasos. v5 = v4 + velocity lineal + contact + fase + limite 0.15.
    - VELOCITY LINEAL MSE(delta_pred, delta_gt) 0.2: reemplaza al log1p (el log1p
      amplifica ruido en trote; sin evidencia publicada para velocidad de joints).
    - DVEL SECUENCIAL (siMLPe dloss, opt-in con --w-dvel): MSE entre diffs de la
      secuencia predicha apilada y diffs del GT. Difiere del vel por paso: ese
      mide el delta contra el carry (que con sampling puede ser prediccion
      propia); este mide la velocidad del movimiento predicho sin importar el
      carry. Ataca P1 directo (fidelidad de paso en el tiempo).
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
    pose_steps = []  # error de pose por paso (para el correctivo leve)
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
        # Con reparam, s0 es q0 original (bloque 0 intacto); h0 siempre vale.
        h0 = state[:, :92].reshape(-1, n_joints, 4)
        if arch == "mono":
            d95 = model(state)
            if delta_std is not None:
                d95 = torch.cat([d95[:, :92] * delta_std, d95[:, 92:95]], dim=1)
            if root_net is not None:
                d95 = torch.cat([d95[:, :92], root_net(state)], dim=1)
            full, root_raw = predict_next(d95, h0, root_prev, root_std)
        else:
            lower, upper, _ = model(state)
            full, root_raw, _ = assemble_prediction(lower, upper, h0, root_prev,
                                                    inferior_idx, superior_idx,
                                                    root_mean, root_std, n_joints)
        true_full = true_win[:, k, :92].reshape(-1, n_joints, 4)
        true_root = true_root_win[:, k]
        wk = w_step1 if k == 0 else 1.0  # Q3: peso extra al paso con input GT
        if loss_shape == "phuber":
            pose_step = quaternion_phuber_loss(full, true_full, phuber_eps)
        else:
            pose_step = quaternion_sign_invariant_loss(full, true_full)
        pose_sum = pose_sum + wk * pose_step
        pose_steps.append(pose_step)
        root_sum = root_sum + wk * ((((root_raw - true_root) / root_std).pow(2)).mean())
        if foot_mode == "batch":
            foot_sum = foot_sum + wk * foot_sliding_loss(full, root_raw, offsets_t, parents,
                                                         (17, 21), (18, 22))
        limit_sum = limit_sum + wk * joint_limit_loss(full, euler_min, euler_max)
        # Velocity LINEAL sobre deltas (v5): MSE directo, sin log1p ni denominador.
        true_delta = true_full - prev_true_win[:, k]
        vel_sum = vel_sum + wk * (((full - h0) - true_delta).pow(2).mean())
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
        if reparam:
            hist_new = reparam_next_state(carry_q.reshape(batch, -1), state, dq_std, batch)
        else:
            hist_new = torch.cat([carry_q.reshape(batch, -1), state[:, :184]], dim=1)
        state = torch.cat([hist_new, root_n, vel_n, feet_n, phase_next], dim=1)
        root_prev = root_raw_c
        if detach_carry:
            state = state.detach()
            root_prev = root_prev.detach()
    stacked = torch.stack(stacked, dim=1)  # (B, K, J, 4)
    # L_CORR leve (correctivo estilo BAgger simplificado): castiga que el error de
    # pose CREZCA de un paso al siguiente (derivar sin volver). Con BPTT truncado
    # cada paso ya es contexto propio congelado; esto presiona a no derivar mas,
    # o sea a volver o sostener. La version agresiva (segundo unroll desde el
    # paso de deriva con loss solo en la vuelta) queda para despues si esta da senal.
    if steps >= 2 and len(pose_steps) == steps:
        esteps = torch.stack(pose_steps)
        corr = torch.relu(esteps[1:] - esteps[:-1]).mean()
    else:
        corr = stacked.sum() * 0.0
    # DVEL secuencial (siMLPe dloss, archivado: inerte a cualquier escala).
    gt_seq = true_win[:, :, :92].reshape(true_win.shape[0], steps, n_joints, 4)
    if steps >= 2:
        dvel = torch.diff(stacked, dim=1).sub(torch.diff(gt_seq, dim=1)).pow(2).mean()
    else:
        dvel = stacked.sum() * 0.0
    # Suavidad sobre el eje TIEMPO (dim=1), no sobre el batch.
    if steps >= 3:
        vel = torch.diff(stacked, dim=1)
        smooth = torch.diff(vel, dim=1).pow(2).mean()
    else:
        smooth = stacked.sum() * 0.0
    # Contact sobre pies predichos apilados (con gradiente) + etiquetas GT.
    pred_feet_seq = torch.stack(pred_feet, dim=1)  # (B, K, 2, 3)
    if steps >= 2:
        contact = foot_contact_loss(pred_feet_seq, contact_mask,
                                    shape=foot_shape, eps=phuber_eps)
    else:
        contact = pred_feet_seq.sum() * 0.0
    wsum = float(steps - 1) + w_step1
    if foot_mode == "time":
        # Foot sliding sobre el eje TIEMPO de la ventana (misma formula que
        # foot_sliding_loss pero dim=1 con minimo por ventana): vive con
        # shuffle (el batch-dim muere: 0.0 exacto con pies flotando).
        fh = pred_feet_seq[:, :, :, 1]
        cmask = fh < (fh.amin(dim=1, keepdim=True) + 0.05)
        move = torch.sqrt(torch.diff(pred_feet_seq[:, :, :, [0, 2]], dim=1).pow(2).sum(dim=-1) + 1e-8)
        pairs = cmask[:, :-1] & cmask[:, 1:]
        foot = (move * pairs.float()).sum() / pairs.float().sum().clamp_min(1.0)
        if pairs.float().sum() == 0:
            foot = move.mean() * 0.0
    else:
        foot = foot_sum / wsum
    return {"pose": pose_sum / wsum, "root": root_sum / wsum, "foot": foot,
            "limit": limit_sum / wsum, "vel": vel_sum / wsum, "contact": contact,
            "smooth": smooth, "dvel": dvel, "corr": corr, "stacked": stacked}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Dispositivo: {device} | unroll: {args.unroll} | vel: {args.w_vel} "
          f"contact: {args.w_contact} limit: {args.w_limit}")

    bundle = np.load(args.data)
    inputs = torch.from_numpy(bundle["inputs"])
    targets = torch.from_numpy(bundle["targets"])
    train_end, val_end = int(bundle["train_end"]), int(bundle["val_end"])
    frame_time = float(bundle["frame_time"])
    phase_period = float(bundle["phase_period"])
    # Multi-clip (npz v2): clip_id por par + periodo por clip. Sin ellos (v1),
    # todo es un solo clip y el regimen queda intacto.
    clip_id = np.asarray(bundle["clip_id"]) if "clip_id" in bundle else None
    clip_periods = np.asarray(bundle["phase_periods"], dtype=np.float64) if "phase_periods" in bundle else None
    pair_periods = clip_periods[clip_id].astype(np.float32) if clip_id is not None else None
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
    # Multi-clip: la ventana [s-1, s+unroll) nunca cruza de clip.
    train_starts = clip_valid_starts(clip_id, 1, train_end - args.unroll + 1, args.unroll)
    val_starts = clip_valid_starts(clip_id, train_end, val_end - args.unroll + 1, args.unroll)
    train_loader = DataLoader(TensorDataset(train_starts), batch_size=args.batch, shuffle=True)
    val_state0 = inputs[val_starts]
    val_true, val_prev = build_windows(targets, val_starts, args.unroll)
    val_root_prev = (root_mean + targets[val_starts - 1][:, 92:95] * root_std)
    val_true_root = val_true[:, :, 92:95] * root_std + root_mean  # metros

    model = (MonoFiLM() if (args.arch == "mono" and args.film) else
             MonoNet() if args.arch == "mono" else BiMoE()).to(device)
    root_net = None
    if args.root_net > 0 and args.arch == "mono":
        root_net = RootNet(hidden=args.root_net).to(device)
    n_params = model.count_parameters() + (root_net.count_parameters() if root_net else 0)
    prefix = "film1x" if (args.arch == "mono" and args.film) else "mono1x" if args.arch == "mono" else "bimoe10k"
    run_id = args.run_id or prefix
    print(f"Arquitectura: {args.arch}{' + root-net' if root_net else ''} | Parametros: {n_params}")
    params = list(model.parameters()) + (list(root_net.parameters()) if root_net else [])
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_final)
    # Apuesta 3: stats de reparam/delta-scale (data, sin params que aprender).
    use_reparam = (args.reparam == "dq" and args.arch == "mono")
    dq_std, delta_std = None, None
    if use_reparam:
        with torch.no_grad():
            hist = inputs[:train_end, :276].reshape(-1, 3, 92)
            dq = (hist[:, 0] - hist[:, 1])
            dq_std = dq.std(dim=0, keepdim=True).clamp_min(1e-4).to(dev)
            dseq = targets[:train_end, :92].reshape(-1, 92)
            delta_std = (dseq[1:] - dseq[:-1]).std(dim=0, keepdim=True).clamp_min(1e-4).to(dev)
        print(f"  [reparam] dq_std~{dq_std.mean().item():.5f} delta_std~{delta_std.mean().item():.5f}")
    # Apuesta 1: EMA shadow (solo inferencia/val, no toca el entreno).
    ema_model, ema_state = None, None
    if args.ema > 0:
        import copy
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)

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
    config = {"arch": args.arch, "film": args.film, "run_id": run_id, "unroll": args.unroll,
              "batch": args.batch, "noise": args.noise, "noise_std": args.noise_std,
              "w_vel": args.w_vel, "w_contact": args.w_contact,
              "w_foot": args.w_foot, "w_limit": args.w_limit,
              "w_dvel": args.w_dvel, "w_root": args.w_root, "w_corr": args.w_corr,
              "ss_end": args.ss_end, "ss_ramp": args.ss_ramp,
              "full_bptt": args.full_bptt, "val_every": args.val_every,
              "min_epochs": args.min_epochs, "patience": args.patience,
              "lr_final": args.lr_final, "ema": args.ema,
              "reparam": args.reparam, "root_net": args.root_net,
              "loss_shape": args.loss_shape, "phuber_eps": args.phuber_eps,
              "foot_shape": args.foot_shape, "w_step1": args.w_step1,
              "foot_mode": args.foot_mode}
    start_epoch, best_val = 0, float("inf")
    best_saved = False  # este run_id solo tiene _best.pt si mejora al heredado
    if args.resume:
        start_epoch, best_val, ema_state = load_resume(args.resume, model, optimizer,
                                                       scheduler, root_net)
        if ema_model is not None:
            if ema_state is not None:
                ema_model.load_state_dict(ema_state)
            else:
                # best.pt guarda pesos EMA como modelo: arrancar la sombra ahi.
                ema_model.load_state_dict(model.state_dict())
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
            "limit": float("nan"), "ref": float("nan"), "roll": float("nan"),
            "dvel": float("nan"), "corr": float("nan")}
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
    # Periodo por muestra (multi-clip) o escalar (v1). La fase avanza
    # 1/periodo del clip propio; con v1 el tensor es el escalar de siempre.
    if pair_periods is not None:
        pp_all = torch.from_numpy(pair_periods)
        def pp_of(starts_t):
            return pp_all[starts_t].to(dev).unsqueeze(1)
    else:
        def pp_of(starts_t):
            return phase_period
    # Seeds del rollout-log dentro de clips de test con hueco para 120f.
    if clip_id is not None:
        bounds = np.asarray(bundle["clip_bounds"])
        test_clips = [i for i in range(len(bounds))
                      if bounds[i, 0] >= val_end and bounds[i, 1] - bounds[i, 0] >= 140]
        roll_seeds, seed_lim, seed_per = [], {}, {}
        per = max(1, args.rollout_seeds) // max(1, len(test_clips)) + 1
        for ci in test_clips:
            lo, hi = int(bounds[ci, 0]), int(bounds[ci, 1])
            for s in np.linspace(lo + 10, hi - 130, num=min(per, 5)).astype(int):
                if len(roll_seeds) >= max(1, args.rollout_seeds):
                    break
                roll_seeds.append(int(s))
                seed_lim[int(s)] = hi - 1
                seed_per[int(s)] = float(clip_periods[int(clip_id[lo])])
            if len(roll_seeds) >= max(1, args.rollout_seeds):
                break
    else:
        roll_seeds = [min(val_end + 10 + i * 80, n_targets - 130)
                      for i in range(max(1, args.rollout_seeds))]
        seed_lim, seed_per = {}, {}

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        # Scheduled sampling: rampa lineal 0 -> ss_end (estados propios utiles,
        # nunca basura pura al inicio ni 100% autoalimentado con 1 clip).
        p_pred = args.ss_end * min(1.0, epoch / max(1, args.ss_ramp))
        for (starts,) in train_loader:
            state0 = inputs[starts].to(dev)
            if use_reparam:
                state0 = reparam_state(state0, dq_std)
            noise_sigma = args.noise_std if args.noise_std > 0 else args.noise
            if noise_sigma > 0:
                noisy = state0.clone()
                noisy[:, :276] += torch.randn_like(noisy[:, :276]) * noise_sigma
                state0 = noisy
            true_win, prev_win = build_windows(targets, starts, args.unroll)
            true_win, prev_win = true_win.to(dev), prev_win.to(dev)
            true_root_win = true_win[:, :, 92:95] * root_std.to(dev) + root_mean.to(dev)
            root_prev = (root_mean.to(dev) + targets[starts - 1][:, 92:95].to(dev) * root_std.to(dev))
            out = unroll(model, state0, root_prev, true_win, prev_win, true_root_win,
                         off_d, parents,
                         inferior_idx, superior_idx, root_mean.to(dev), root_std.to(dev),
                         vel_mean.to(dev), vel_std.to(dev), feet_mean.to(dev), feet_std.to(dev),
                         euler_min.to(dev), euler_max.to(dev), frame_time, pp_of(starts),
                         w_vel=args.w_vel, w_contact=args.w_contact,
                         w_foot=args.w_foot, w_limit=args.w_limit,
                         w_dvel=args.w_dvel,
                         p_pred=p_pred, detach_carry=detach_carry, arch=args.arch,
                         root_net=root_net, loss_shape=args.loss_shape,
                         phuber_eps=args.phuber_eps, foot_shape=args.foot_shape,
                         w_step1=args.w_step1, reparam=use_reparam, dq_std=dq_std,
                         delta_std=delta_std, foot_mode=args.foot_mode)
            reg = sum(p.pow(2).sum() for p in params)
            loss = (out["pose"] + args.w_root * out["root"] + args.w_foot * out["foot"]
                    + 0.3 * out["smooth"] + args.w_limit * out["limit"]
                    + args.w_vel * out["vel"] + args.w_contact * out["contact"]
                    + args.w_dvel * out["dvel"] + args.w_corr * out["corr"]
                    + 1e-5 * reg)
            if not torch.isfinite(loss):
                with torch.no_grad():
                    diag = {k: (v.item() if torch.is_tensor(v) else v)
                            for k, v in {"pose": out["pose"], "root": out["root"],
                                         "foot": out["foot"], "contact": out["contact"],
                                         "smooth": out["smooth"], "limit": out["limit"],
                                         "vel": out["vel"], "dvel": out["dvel"], "corr": out["corr"]}.items()}
                print(f"  [epoch {epoch}] batch no-finito: {diag}", flush=True)
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            # Anti-envenenamiento: un grad inf (p.ej. asin en el borde x8 BPTT)
            # vuelve NaN el clip (inf*0) y mata los pesos para siempre.
            if not all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in params):
                print(f"  [epoch {epoch}] grad no-finito, se salta", flush=True)
                optimizer.zero_grad()
                continue
            optimizer.step()
            if ema_model is not None:
                with torch.no_grad():
                    for e, m in zip(ema_model.parameters(), model.parameters()):
                        e.mul_(args.ema).add_(m.detach(), alpha=1.0 - args.ema)
            train_loss += loss.item()
        scheduler.step()

        # Val con cadencia (el grueso del costo por epoch): ep1, cada val_every y ultima.
        do_val = (epoch == 1) or (epoch % val_every == 0) or (epoch == args.epochs)
        val_finite = False
        if do_val:
            if ema_model is not None:
                ema_model.eval()
                eval_model, eval_state = ema_model, ema_model.state_dict()
            else:
                model.eval()
                eval_model, eval_state = model, None
            with torch.no_grad():
                # Val con el MISMO p_pred final (mide el regimen del juego).
                vs0 = val_state0.to(dev)
                if use_reparam:
                    vs0 = reparam_state(vs0, dq_std)
                v = unroll(eval_model, vs0, val_root_prev.to(dev),
                           val_true.to(dev), val_prev.to(dev), val_true_root.to(dev),
                           off_d, parents,
                           inferior_idx, superior_idx, root_mean.to(dev), root_std.to(dev),
                           vel_mean.to(dev), vel_std.to(dev), feet_mean.to(dev), feet_std.to(dev),
                           euler_min.to(dev), euler_max.to(dev), frame_time, pp_of(val_starts),
                           w_vel=args.w_vel, w_contact=args.w_contact,
                           w_foot=args.w_foot, w_limit=args.w_limit,
                           w_dvel=args.w_dvel,
                           p_pred=args.ss_end, arch=args.arch,
                           root_net=root_net, loss_shape=args.loss_shape,
                           phuber_eps=args.phuber_eps, foot_shape=args.foot_shape,
                           w_step1=args.w_step1, reparam=use_reparam, dq_std=dq_std,
                           delta_std=delta_std, foot_mode=args.foot_mode)
                val_tensor = (v["pose"] + args.w_root * v["root"] + args.w_foot * v["foot"]
                              + args.w_limit * v["limit"] + args.w_vel * v["vel"]
                              + args.w_contact * v["contact"]
                              + args.w_dvel * v["dvel"] + args.w_corr * v["corr"])
                val_finite = bool(torch.isfinite(val_tensor).item())
                val_loss = val_tensor.item() if val_finite else float("nan")
                # Referencia one-step clasica sobre pares de val (base real -> target).
                if args.arch == "mono":
                    rs0 = val_state0[:512].to(dev)
                    if use_reparam:
                        rs0 = reparam_state(rs0, dq_std)
                    d95 = eval_model(rs0)
                    if delta_std is not None:
                        d95 = torch.cat([d95[:, :92] * delta_std, d95[:, 92:95]], dim=1)
                    if root_net is not None:
                        d95 = torch.cat([d95[:, :92], root_net(rs0)], dim=1)
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
                                 "limit": v["limit"].item(), "ref": ref.item(),
                                 "dvel": v["dvel"].item(), "corr": v["corr"].item()})
            # Rollout 120f liviano para el log (solo en epochs de val).
            if epoch % max(1, args.rollout_every) == 0:
                with torch.no_grad():
                    last["roll"] = rollout_120f_metric(eval_model if ema_model is not None else model, args.arch, tq_all, tr_all,
                                                       phase_all, phase_period, roll_seeds,
                                                       stats_d, off_d, parents,
                                                       inferior_idx, superior_idx,
                                                       frame_time, root_net=root_net,
                                                       reparam=use_reparam, dq_std=dq_std,
                                                       delta_std=delta_std,
                                                       seed_lim=seed_lim, seed_per=seed_per)
        tag = "" if do_val else " [sin val]"
        line = (f"epoch {epoch:3d} train {train_loss / len(train_loader):.6f} "
                f"val_unroll {last['val']:.6f} (pose {last['pose']:.6f} "
                f"root {last['root']:.6f} vel {last['vel']:.6f} "
                f"contact {last['contact']:.6f} foot {last['foot']:.6f} "
                f"limit {last['limit']:.6f} dvel {last['dvel']:.6f} corr {last['corr']:.6f}) ref1 {last['ref']:.6f} "
                f"roll120_feet {last['roll']:.3f} p {p_pred:.2f}{tag}")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

        if do_val:
            if val_finite and val_loss < best_val:
                best_val = val_loss
                best_saved = True
                patience_left = args.patience
                save_checkpoint(out_dir / f"{run_id}_best.pt", epoch,
                                ema_model if ema_model is not None else model,
                                optimizer, scheduler, best_val, config,
                                ema_state=None,
                                root_state=(root_net.state_dict() if root_net else None))
            else:
                # Cada chequeo fallido descuenta val_every epochs (regla: 10 epochs).
                patience_left -= val_every
        if epoch % max(1, args.ckpt_every) == 0:
            save_checkpoint(out_dir / f"{run_id}_epoch{epoch}_pose{last['pose']:.4f}_foot{last['foot']:.4f}.pt",
                            epoch, model, optimizer, scheduler, best_val, config,
                            ema_state=(ema_model.state_dict() if ema_model else None),
                            root_state=(root_net.state_dict() if root_net else None))
        # El corte VA DESPUES del save: el ultimo epoch siempre queda en disco.
        if do_val and patience_left <= 0 and epoch >= args.min_epochs:
            print("Early stopping.")
            break
    if best_saved:
        print(f"Mejor val_unroll: {best_val:.6f} -> {out_dir / f'{run_id}_best.pt'}")
    else:
        print(f"Sin mejora en este run (mejor heredado {best_val:.6f}): "
              f"no hay {out_dir / f'{run_id}_best.pt'}; el mejor sigue en el ckpt de --resume")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena BiMoE-10K v3 (desenrollado)")
    parser.add_argument("--arch", default="moe", choices=["moe", "mono"],
                        help="moe = router+2 expertos (BiMoE), mono = una sola red (1X)")
    parser.add_argument("--film", action="store_true",
                        help="mono con fase que modula ocultas (vaso nuevo PFNN-barato; NO admite --resume de pesos mono, run fresco)")
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
    parser.add_argument("--w-dvel", type=float, default=0.0,
                        help="peso velocity SECUENCIAL siMLPe (diffs de secuencia predicha vs GT; 0 = off, regimen intacto)")
    parser.add_argument("--w-root", type=float, default=1.0,
                        help="peso root (1.0 = regimen intacto; >1 fuerza control de trayectoria)")
    parser.add_argument("--w-corr", type=float, default=0.0,
                        help="peso correctivo leve: castiga crecimiento del error de pose paso a paso (0 = off)")
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
    parser.add_argument("--lr-final", type=float, default=0.0,
                        help="apuesta 1: lr final del coseno (0 = a cero, regimen intacto; 1e-6 = anneal)")
    parser.add_argument("--ema", type=float, default=0.0,
                        help="apuesta 1: sombra EMA con este decay (0 = off; 0.999 = promedia jitter Adam)")
    parser.add_argument("--reparam", default="none", choices=["none", "dq"],
                        help="apuesta 3: input [q0,dq1,dq2] z-scored + output delta x delta_std (none = intacto, solo mono)")
    parser.add_argument("--root-net", type=int, default=0,
                        help="apuesta 2: cabeza root aislada con este hidden (0 = off, solo mono; 32 = ~1.5k params)")
    parser.add_argument("--loss-shape", default="mse", choices=["mse", "phuber"],
                        help="apuesta 5: forma de la loss de pose (mse = intacto)")
    parser.add_argument("--phuber-eps", type=float, default=0.003,
                        help="apuesta 5: eps pseudo-Huber (~0.5*RMS1, escala del delta por joint)")
    parser.add_argument("--foot-shape", default="mse", choices=["mse", "sqrt", "phuber"],
                        help="apuesta 5: forma del contact (mse = cuadrado intacto; sqrt = L2 MoDi, grad 55-200x)")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="apuesta 4: sigma GNS en historial quats (0 = usa --noise; ~0.5*RMS1)")
    parser.add_argument("--w-step1", type=float, default=1.0,
                        help="Q3: peso del paso 0 del unroll (input GT puro; 1.0 = intacto, 3.0 = ataca el intercepto)")
    parser.add_argument("--foot-mode", default="batch", choices=["batch", "time"],
                        help="foot sliding en dim batch (v5, muere con shuffle) o time (ventana, vive siempre)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
