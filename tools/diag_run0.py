#!/usr/bin/env python3
"""Run 0 CPU del analisis (M1-M7): diagnostico antes de gastar un run de GPU.

Mide, por checkpoint (default: film1_best, wroot4_ep70, pose_best):
  M1 P1 en h=1,2,3,5 + rootMSE por horizonte (root_share = a_root/a)
  M2 P1(h=1) vs ref1 en las mismas unidades (difiere >20% = bug de harness)
  M3 baseline velocidad-constante CV en P1 (CV(5f)<=0.002 = ruido/sesgo, no capacidad)
  M4 a en ventanas TRAIN vs TEST (a_tr/a_te>0.7 = fit; <0.4 = generalizacion)
  M5 SNR1 teacher-forced en val (E||d||^2/E||d-dhat||^2; <10 = probar reparam)
  M6 pies sobre el clip GT con la misma funcion (piso de la metrica)
  M7 58 ventanas stride 10 + bootstrap por bloques de 9 (IC90 + cobertura de fase)

Uso: .venv/bin/python tools/diag_run0.py --ckpt /tmp/opencode/ckpts/v5_film1_best.pt \
       --ckpt /tmp/opencode/ckpts/v5_mono_wroot4_epoch70_pose0.0097_foot0.0420.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from eval_gates import load_all, rollout, quat_mse, cv_baseline  # noqa: E402
from mono_model import predict_next  # noqa: E402
from losses import quaternion_sign_invariant_loss, forward_kinematics_batch  # noqa: E402


def p1_for(model, arch, tq, tr, phase, period, seeds, h, stats, off, parents, dt,
           inf_idx, sup_idx, root_net=None, reparam=False, dq_std=None, delta_std=None,
           with_cv=False):
    ms, cs = [], []
    for s in seeds:
        pred, _ = rollout(model, arch, tq, tr, phase, period, s, h, stats, off,
                          parents, dt, inf_idx, sup_idx, root_net=root_net,
                          reparam=reparam, dq_std=dq_std, delta_std=delta_std)
        real = tq[s + 1:s + 1 + h].numpy()
        ms.append(quat_mse(pred, real[:len(pred)]))
        if with_cv:
            cs.append(quat_mse(cv_baseline(tq, s, h), real[:h]))
    return float(np.mean(ms)), (float(np.mean(cs)) if cs else 0.0), ms


def p1_free_root(model, tq, tr, phase, period, seeds, h, stats, off, parents, dt,
                 root_net=None, reparam=False, dq_std=None, delta_std=None, **_):
    """M1: P1 con root PROPIO integrado (sin velocidad real forzada).
    La brecha free-vs-forced = cuanto del drift maneja el root (apuesta 2)."""
    rm, rs, vm, vs, fm, fs = (stats["root_mean"], stats["root_std"], stats["vel_mean"],
                              stats["vel_std"], stats["feet_mean"], stats["feet_std"])
    errs = []
    with torch.no_grad():
        for s in seeds:
            hist = [tq[s].clone(), tq[s - 1].clone(), tq[s - 2].clone()]
            root = tr[s].copy()
            root_prev = root.copy()
            ph = float(phase[s + 3] % 1.0)
            pred = []
            for _ in range(h):
                root_t = torch.from_numpy(root).float()
                vel_t = torch.from_numpy((root - root_prev) / dt).float()
                pos = forward_kinematics_batch(hist[0].reshape(1, 23, 4).float(),
                                               root_t.reshape(1, 3), off, parents)
                feet = torch.stack([torch.minimum(pos[0, 21], pos[0, 22]),
                                    torch.minimum(pos[0, 17], pos[0, 18])])[:, 1]
                st = torch.cat([x.reshape(-1) for x in hist] +
                               [(root_t - rm) / rs, (vel_t - vm) / vs, (feet - fm) / fs,
                                torch.tensor([ph])])
                if reparam and dq_std is not None:
                    q0, q1, q2 = st[:92], st[92:184], st[184:276]
                    st = torch.cat([q0, (q0 - q1) / dq_std.squeeze(0),
                                    (q1 - q2) / dq_std.squeeze(0), st[276:]])
                d95 = model(st.unsqueeze(0))
                if delta_std is not None:
                    d95 = torch.cat([d95[:, :92] * delta_std, d95[:, 92:95]], dim=1)
                if root_net is not None:
                    d95 = torch.cat([d95[:, :92], root_net(st.unsqueeze(0))], dim=1)
                nxt, root_raw = predict_next(d95, hist[0].reshape(1, 23, 4),
                                             root_t.reshape(1, 3), rs)
                nxt = nxt.reshape(23, 4)
                hist = [nxt, hist[0], hist[1]]
                root_prev = root.copy()
                root = root_raw.reshape(3).numpy()
                ph = (ph + 1.0 / period) % 1.0
                pred.append(nxt.numpy())
            real = tq[s + 1:s + 1 + h].numpy()
            errs.append(quat_mse(np.array(pred), real[:len(pred)]))
    return float(np.mean(errs))


def main():
    a = parse_args()
    out = {}
    for ckpt_path in a.ckpt:
        print(f"=== {Path(ckpt_path).name} ===")
        (model, bundle, parents, off, dt, stats, tq, tr, phase, period, arch,
         root_net, reparam, dq_std, delta_std) = load_all(ckpt_path)
        inf_idx = sorted(set(int(j) for j in bundle["inferior_joints"]))
        sup_idx = sorted(set(int(j) for j in bundle["superior_joints"]))
        te, va, n = int(bundle["train_end"]), int(bundle["val_end"]), len(tq)
        seeds8 = sorted(set(min(va + 10 + i * 80, n - 130) for i in range(8)))
        kw = dict(stats=stats, off=off, parents=parents, dt=dt, inf_idx=inf_idx,
                  sup_idx=sup_idx, root_net=root_net, reparam=reparam,
                  dq_std=dq_std, delta_std=delta_std)
        # M1: horizontes cortos + CV + root propio (particion del intercepto).
        hs = [1, 2, 3, 5, 10, 20, 24]
        p1 = {h: p1_for(model, arch, tq, tr, phase, period, seeds8, h, **kw,
                        with_cv=True) for h in hs}
        for h in hs:
            print(f"  P1({h:2d}) pose {p1[h][0]:.5f} cv {p1[h][1]:.5f}")
        free5 = p1_free_root(model, tq, tr, phase, period, seeds8, 5, **kw)
        free24 = p1_free_root(model, tq, tr, phase, period, seeds8, 24, **kw)
        share = (free24 - p1[24][0]) / max(free24, 1e-12)
        print(f"  M1 free-root P1_5 {free5:.5f} P1_24 {free24:.5f} share {share:.2f} "
              f"(>=0.5 -> apuesta 2; <0.3 -> descartarla)")
        # Ajuste a+ b*h con 5f y 24f + chequeo 10/20.
        b = (p1[24][0] - p1[5][0]) / 19.0
        a0 = p1[5][0] - 5 * b
        chk10 = abs((a0 + 10 * b) - p1[10][0]) / p1[10][0]
        chk20 = abs((a0 + 20 * b) - p1[20][0]) / p1[20][0]
        print(f"  a={a0:.5f} b={b:.6f}/f chequeo 10f {chk10:.1%} 20f {chk20:.1%}")
        # M2: ref1 teacher-forced vs P1(h=1), mismas unidades.
        with torch.no_grad():
            vs = bundle["inputs"][va:va + 512].astype(np.float32)
            vs_t = torch.from_numpy(vs)
            if reparam and dq_std is not None:
                q0, q1, q2 = vs_t[:, :92], vs_t[:, 92:184], vs_t[:, 184:276]
                vs_t = torch.cat([q0, (q0 - q1) / dq_std, (q1 - q2) / dq_std,
                                  vs_t[:, 276:]], dim=1)
            d95 = model(vs_t)
            if delta_std is not None:
                d95 = torch.cat([d95[:, :92] * delta_std, d95[:, 92:95]], dim=1)
            if root_net is not None:
                d95 = torch.cat([d95[:, :92], root_net(vs_t)], dim=1)
            bq = torch.from_numpy(vs[:, :92].reshape(-1, 23, 4))
            br = torch.from_numpy(vs[:, 276:279] * bundle["root_std"].astype(np.float32)
                                  + bundle["root_mean"].astype(np.float32))
            fq, _ = predict_next(d95, bq, br, torch.from_numpy(
                bundle["root_std"].astype(np.float32)))
            tq0 = torch.from_numpy(bundle["targets"][va:va + 512, :92].reshape(-1, 23, 4))
            ref1 = float(quaternion_sign_invariant_loss(fq, tq0).item())
        print(f"  M2 ref1 {ref1:.5f} vs P1(1) {p1[1][0]:.5f} "
              f"(dif {abs(ref1 - p1[1][0]) / max(p1[1][0], 1e-9):.1%})")
        # M3: CV baseline.
        cvs = {}
        for h in [5, 10, 20, 24]:
            errs = []
            for s in seeds8:
                real = tq[s + 1:s + 1 + h].numpy()
                errs.append(quat_mse(cv_baseline(tq, s, h), real[:h]))
            cvs[h] = float(np.mean(errs))
        print(f"  M3 CV: " + " ".join(f"{h}f {cvs[h]:.5f}" for h in [5, 10, 20, 24]))
        # M4: a en TRAIN vs TEST.
        tseeds = sorted(set(min(10 + i * 80, te - 130) for i in range(8)))
        tr5, _, _ = p1_for(model, arch, tq, tr, phase, period, tseeds, 5, **kw)
        tr24, _, _ = p1_for(model, arch, tq, tr, phase, period, tseeds, 24, **kw)
        a_tr = tr5 - 5 * (tr24 - tr5) / 19.0
        print(f"  M4 a_train {a_tr:.5f} a_test {a0:.5f} ratio {a_tr / max(a0, 1e-9):.2f}")
        # M5: SNR1 teacher-forced (par M2: senal = ||tq-bq||^2, ruido = sign-inv).
        with torch.no_grad():
            sig = (tq0 - bq).pow(2).sum(dim=-1)
            nse = torch.minimum((fq - tq0).pow(2).sum(dim=-1),
                                (fq + tq0).pow(2).sum(dim=-1))
            snr = float((sig.mean() / nse.mean().clamp_min(1e-12)).item())
        print(f"  M5 SNR1 {snr:.2f} (<10 -> probar reparam)")
        # M6: piso GT de pies.
        with torch.no_grad():
            meds = []
            for s in seeds8:
                q = torch.from_numpy(bundle["targets"][s:s + 120, :92].astype(
                    np.float32)).reshape(-1, 23, 4)
                r = torch.from_numpy((bundle["targets"][s:s + 120, 92:95].astype(
                    np.float32) * bundle["root_std"].astype(np.float32)
                    + bundle["root_mean"].astype(np.float32)))
                pos = forward_kinematics_batch(q, r, off, parents).numpy()
                feet = np.minimum(np.minimum(pos[:, 21, 1], pos[:, 22, 1]),
                                  np.minimum(pos[:, 17, 1], pos[:, 18, 1]))
                meds.append(float(np.median(feet)))
        print(f"  M6 pies_GT {np.median(meds):.3f}m (piso de la metrica)")
        # M7: 58 ventanas stride 10 + bootstrap por bloques de 9 + cobertura fase.
        w58 = []
        for i in range(58):
            s = min(va + 10 + i * 10, n - 130)
            if s not in w58:
                w58.append(s)
        nw = len(w58)
        per5 = [p1_for(model, arch, tq, tr, phase, period, [s], 5, **kw)[0] for s in w58]
        per24 = [p1_for(model, arch, tq, tr, phase, period, [s], 24, **kw)[0] for s in w58]
        rng = np.random.default_rng(0)
        bmeans = []
        for _ in range(2000):
            idx = np.concatenate([np.arange(s, s + 9) % nw
                                  for s in rng.integers(0, nw, size=max(1, nw // 9))])[:nw]
            bmeans.append(np.mean([per24[i] for i in idx]))
        lo, hi = float(np.quantile(bmeans, 0.05)), float(np.quantile(bmeans, 0.95))
        ph_cov = sorted(float(phase[s + 3] % 1.0) for s in w58)
        gaps = max(np.diff(ph_cov).tolist() + [ph_cov[0] + 1 - ph_cov[-1]])
        print(f"  M7 P1_24 media {np.mean(per24):.5f} IC90 [{lo:.5f},{hi:.5f}] "
              f"max-gap-fase {gaps:.3f} (8-seeds cubre ~0.52 del ciclo)")
        out[Path(ckpt_path).name] = {
            "p1": {h: v[0] for h, v in p1.items()}, "a": a0, "b": b,
            "ref1": ref1, "cv": cvs, "a_train": a_tr, "snr1": snr,
            "pies_gt": float(np.median(meds)), "p1_24_ic90": [lo, hi],
        }
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"JSON -> {a.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Run 0 CPU: M1-M7 antes de gastar GPU")
    p.add_argument("--ckpt", action="append", required=True)
    p.add_argument("--out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()
