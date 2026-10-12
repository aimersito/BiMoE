#!/usr/bin/env python3
"""Demo interactivo BiMoE-10K: esqueleto con GLFW/OpenGL controlado con WASD.

  W/S: avanzar / retroceder.  A/D: girar.  Espacio: frenar.  Esc: salir.
  Boton derecho: orbitar.  Rueda: zoom.
  El teclado mueve la raiz; el modelo genera la pose cada frame.

Uso:
  .venv/bin/python src/play.py [--walk] [--frames=N] [--shot=img.png:frame]
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import torch

from bimoe_model import BiMoE
from losses import forward_kinematics_batch
from preprocess import parse_bvh

import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

WALK_SPEED = None  # se mide del BVH al arrancar


class GameState:
    """Estado del personaje: pose + raiz + velocidad + giro (yaw)."""

    def __init__(self, quats, root, prev_quats=None, prev2_quats=None,
                 velocity=None, yaw=0.0):
        self.quats = quats                  # (23, 4) torch, frame t
        # Historial con frames reales consecutivos (no clones: el clon
        # estatico es OOD y mete un transitorio que realimenta el error).
        self.prev_quats = prev_quats.clone() if prev_quats is not None else quats.clone()
        self.prev2_quats = prev2_quats.clone() if prev2_quats is not None else quats.clone()
        self.root = root.astype(np.float64)  # (3,) metros
        self.yaw = yaw
        # Velocidad inicial = velocidad real del seed (no cero: cero tambien es OOD).
        self.velocity = np.array(velocity, dtype=np.float64) if velocity is not None else np.zeros(3)
        self.keys = set()

    def desired_velocity(self):
        forward = np.array([np.cos(self.yaw), 0.0, np.sin(self.yaw)])
        speed = 0.0
        if "w" in self.keys:
            speed += WALK_SPEED
        if "s" in self.keys:
            speed -= WALK_SPEED * 0.6
        return forward * speed

    def step_turn(self, dt):
        if "a" in self.keys:
            self.yaw += 1.8 * dt
        if "d" in self.keys:
            self.yaw -= 1.8 * dt


def build_state(game, vel_mean, vel_std, offsets_t, parents, root_mean, root_std, feet_mean, feet_std):
    """Vector de entrada de 284 numeros desde el estado actual del juego.
    OJO: vel y pies van NORMALIZADOS con las stats crudas del entreno,
    igual que en training (antes se pasaban en crudo: bug OOD)."""
    with torch.no_grad():
        pos = forward_kinematics_batch(game.quats.reshape(1, 23, 4).float(),
                                       torch.from_numpy(game.root).float().reshape(1, 3),
                                       offsets_t, parents)
    feet = torch.stack([torch.minimum(pos[0, 21], pos[0, 22]),
                        torch.minimum(pos[0, 17], pos[0, 18])])[:, 1]
    root_t = torch.from_numpy(game.root).float()
    vel_t = torch.from_numpy(game.velocity).float()
    state = torch.cat([game.quats.reshape(-1).float(),
                       game.prev_quats.reshape(-1).float(),
                       game.prev2_quats.reshape(-1).float(),
                       (root_t - root_mean) / root_std,
                       (vel_t - vel_mean) / vel_std,
                       (feet - feet_mean) / feet_std])
    return state


def model_step(model, game, vel_mean, vel_std, offsets_t, parents, root_mean, root_std,
               feet_mean, feet_std, inferior_idx, superior_idx, dt):
    """Un frame: el teclado mueve la raiz, el modelo pone la pose."""
    game.step_turn(dt)
    if "space" in game.keys:
        game.velocity *= max(0.0, 1.0 - 6.0 * dt)
    else:
        desired = game.desired_velocity()
        game.velocity += (desired - game.velocity) * min(1.0, 8.0 * dt)
    game.root += game.velocity * dt
    state = build_state(game, vel_mean, vel_std, offsets_t, parents, root_mean, root_std,
                        feet_mean, feet_std)
    with torch.no_grad():
        lower, upper, _ = model(state)
    delta = torch.zeros(23, 4)
    delta[inferior_idx] = lower[:36].reshape(-1, 4)
    delta[superior_idx] = upper.reshape(-1, 4)
    game.prev2_quats = game.prev_quats
    game.prev_quats = game.quats
    # v4 residual: next = base + delta.
    nxt = game.quats + delta
    game.quats = nxt / nxt.norm(dim=-1, keepdim=True)


class Vista:
    """Ventana GLFW: dibuja piso + esqueleto que genera el modelo."""

    def __init__(self, game, parents, names, w=1280, h=720):
        self.game = game
        self.parents = parents
        self.names = names
        self.held = game.keys
        self.cam_yaw = 0.5
        self.cam_pitch = 0.35
        self.cam_dist = 4.0
        self.rdown = False
        self.last_xy = (0, 0)
        if not glfw.init():
            raise SystemExit("glfw.init fallo (¿DISPLAY?)")
        glfw.window_hint(glfw.SAMPLES, 4)
        self.win = glfw.create_window(w, h, "BiMoE-10K jugable — WASD mueve", None, None)
        if not self.win:
            glfw.terminate()
            raise SystemExit("sin ventana GL")
        glfw.make_context_current(self.win)
        try:
            glfw.swap_interval(1)
        except Exception:
            pass
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glClearColor(0.07, 0.07, 0.07, 1.0)
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_mouse_button_callback(self.win, self.on_mouse)
        glfw.set_cursor_pos_callback(self.win, self.on_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)
        self.fps_t = time.perf_counter()
        self.ivals = []

    def on_key(self, win, key, scode, action, mods):
        name = {glfw.KEY_W: "w", glfw.KEY_A: "a", glfw.KEY_S: "s",
                glfw.KEY_D: "d", glfw.KEY_SPACE: "space"}.get(key)
        if name and action == glfw.PRESS:
            self.held.add(name)
        elif name and action == glfw.RELEASE:
            self.held.discard(name)
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)

    def on_mouse(self, win, btn, action, mods):
        if btn == glfw.MOUSE_BUTTON_RIGHT:
            self.rdown = (action == glfw.PRESS)

    def on_move(self, win, x, y):
        if self.rdown:
            dx, dy = x - self.last_xy[0], y - self.last_xy[1]
            self.cam_yaw += dx * 0.008
            self.cam_pitch = max(-1.4, min(1.4, self.cam_pitch + dy * 0.008))
        self.last_xy = (x, y)

    def on_scroll(self, win, dx, dy):
        self.cam_dist = max(1.0, min(30.0, self.cam_dist * (0.9 ** dy)))

    def frame(self, positions):
        gx, gz = float(self.game.root[0]), float(self.game.root[2])
        w, h = glfw.get_framebuffer_size(self.win)
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(50.0, w / max(h, 1), 0.1, 300.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        tx, tz, ty = gx, gz, 1.0
        cp = math.cos(self.cam_pitch)
        ex = tx + self.cam_dist * math.sin(self.cam_yaw) * cp
        ey = ty + self.cam_dist * math.sin(self.cam_pitch)
        ez = tz + self.cam_dist * math.cos(self.cam_yaw) * cp
        gluLookAt(ex, ey, ez, tx, ty, tz, 0, 1, 0)
        # Piso.
        glLineWidth(1)
        glBegin(GL_LINES)
        glColor3f(0.16, 0.16, 0.16)
        x0, x1 = int(gx) - 12, int(gx) + 12
        z0, z1 = int(gz) - 12, int(gz) + 12
        for i in range(x0, x1 + 1):
            glVertex3f(i, 0, z0)
            glVertex3f(i, 0, z1)
        for i in range(z0, z1 + 1):
            glVertex3f(x0, 0, i)
            glVertex3f(x1, 0, i)
        glEnd()
        # Huesos.
        glLineWidth(3)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.3, 0.2)
        for k, p in enumerate(self.parents):
            if p < 0:
                continue
            glVertex3f(*positions[p])
            glVertex3f(*positions[k])
        glEnd()
        # Articulaciones.
        glPointSize(7)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.3, 0.2)
        for k in range(len(self.parents)):
            glVertex3f(*positions[k])
        glEnd()
        # Direccion deseada.
        vel = self.game.velocity
        if np.linalg.norm(vel) > 0.05:
            d = vel / np.linalg.norm(vel)
            glPointSize(9)
            glBegin(GL_POINTS)
            glColor3f(1.0, 0.92, 0.0)
            v = float(np.linalg.norm(vel))
            for o in (0.2, 0.4, 0.6):
                glVertex3f(gx + d[0] * v * o, 0.05, gz + d[2] * v * o)
            glEnd()
        now = time.perf_counter()
        self.ivals.append(now - self.fps_t)
        if len(self.ivals) > 30:
            self.ivals.pop(0)
        self.fps_t = now
        glfw.swap_buffers(self.win)
        return len(self.ivals) / max(sum(self.ivals), 1e-9)

    def run(self, tick, fps=60, max_frames=0, walk=False, shot=None):
        step = 1.0 / fps
        next_t = time.perf_counter()
        if walk:
            self.held.add("w")
        shot_at = None
        if shot:
            path, at = shot.rsplit(":", 1)
            shot_at = (path, int(at))
        nframes = 0
        last_fps = time.perf_counter()
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            positions = tick(1.0 / fps)
            show_fps = self.frame(positions)
            nframes += 1
            if shot_at and nframes == shot_at[1]:
                from PIL import Image
                w, h = glfw.get_framebuffer_size(self.win)
                px = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
                img = Image.frombytes("RGB", (w, h), px)
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                img.save(shot_at[0])
                print("shot: %s" % shot_at[0], flush=True)
            if max_frames and nframes >= max_frames:
                break
            if time.perf_counter() - last_fps >= 2.0:
                print("FPS: %.1f  teclas: %s" % (show_fps, sorted(self.held)), flush=True)
                last_fps = time.perf_counter()
            next_t += step
            now = time.perf_counter()
            if next_t < now - step:
                next_t = now
            while True:
                remain = next_t - time.perf_counter()
                if remain <= 0:
                    break
                if remain > 0.003:
                    time.sleep(remain - 0.002)
        glfw.terminate()


def main():
    args = parse_args()
    bundle = np.load(ROOT / "data" / "locomotion.npz")
    inputs = torch.from_numpy(bundle["inputs"])
    train_end = int(bundle["train_end"])
    inferior_idx = sorted(set(int(j) for j in bundle["inferior_joints"]))
    superior_idx = sorted(set(int(j) for j in bundle["superior_joints"]))
    root_mean = torch.from_numpy(bundle["root_mean"].astype(np.float32))
    root_std = torch.from_numpy(bundle["root_std"].astype(np.float32))
    # Stats CRUDAS (no las columnas ya normalizadas del npz).
    vel_mean = torch.from_numpy(bundle["vel_mean"].astype(np.float32))
    vel_std = torch.from_numpy(bundle["vel_std"].astype(np.float32))
    feet_mean = torch.from_numpy(bundle["feet_mean"].astype(np.float32))
    feet_std = torch.from_numpy(bundle["feet_std"].astype(np.float32))

    names, parents, offsets, _, motion, frame_time = parse_bvh(
        ROOT / "dataset" / "additional_basic_locomotion_2.bvh")
    offsets_t = torch.tensor(np.array(offsets), dtype=torch.float32)

    global WALK_SPEED
    root_motion = motion[:, :3]
    speeds = np.linalg.norm(np.diff(root_motion[:, [0, 2]], axis=0), axis=1) / frame_time
    WALK_SPEED = float(np.median(speeds))
    print(f"Velocidad de marcha medida: {WALK_SPEED:.2f} m/s", flush=True)

    model = BiMoE()
    model.load_state_dict(torch.load(ROOT / "weights" / "bimoe10k_best.pt", map_location="cpu"))
    model.eval()

    # Seed con 3 frames reales consecutivos + velocidad/yaw reales del clip
    # (targets[i] = frame i+3; seed=val_end -> frames seed, seed+1, seed+2).
    seed = int(bundle["val_end"])
    q_t = torch.from_numpy(bundle["targets"][seed - 1, :92]).reshape(23, 4).float()
    q_t1 = torch.from_numpy(bundle["targets"][seed - 2, :92]).reshape(23, 4).float()
    q_t2 = torch.from_numpy(bundle["targets"][seed - 3, :92]).reshape(23, 4).float()
    seed_root = bundle["root_mean"] + bundle["targets"][seed - 1, 92:95] * bundle["root_std"]
    root_t1 = bundle["root_mean"] + bundle["targets"][seed - 2, 92:95] * bundle["root_std"]
    seed_vel = (np.array(seed_root) - np.array(root_t1)) / frame_time
    seed_yaw = float(np.arctan2(seed_vel[2], seed_vel[0]))
    game = GameState(q_t, np.array(seed_root, dtype=np.float64),
                     prev_quats=q_t1, prev2_quats=q_t2,
                     velocity=seed_vel, yaw=seed_yaw)

    if args.smoke:
        game.keys.add("w")
        for _ in range(120):
            model_step(model, game, vel_mean, vel_std, offsets_t, parents, root_mean, root_std,
                       feet_mean, feet_std, inferior_idx, superior_idx, frame_time)
        assert torch.isfinite(game.quats).all(), "NaN en quats"
        norms = game.quats.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "quats no unitarios"
        moved = np.linalg.norm(game.root - np.array(seed_root))
        print(f"SMOKE OK: camino {moved:.2f} m en 120 frames, sin NaN.")
        return

    vista = Vista(game, parents, names)

    def tick(dt):
        model_step(model, game, vel_mean, vel_std, offsets_t, parents, root_mean, root_std,
                   feet_mean, feet_std, inferior_idx, superior_idx, dt)
        with torch.no_grad():
            pos = forward_kinematics_batch(game.quats.reshape(1, 23, 4).float(),
                                           torch.from_numpy(game.root).float().reshape(1, 3),
                                           offsets_t, parents)[0].numpy()
        return pos

    vista.run(tick, max_frames=args.frames, walk=args.walk, shot=args.shot)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo jugable BiMoE-10K")
    parser.add_argument("--smoke", action="store_true", help="prueba sin ventana: 120 frames con w")
    parser.add_argument("--walk", action="store_true", help="camina solo al arrancar")
    parser.add_argument("--frames", type=int, default=0, help="cierra tras N frames (0 = infinito)")
    parser.add_argument("--shot", default=None, help=" captura: ruta.png:frame")
    return parser.parse_args()


if __name__ == "__main__":
    main()
