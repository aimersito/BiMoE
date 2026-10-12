#!/usr/bin/env python3
"""Preproceso BiMoE-10K: BVH -> pares (estado_t -> estado_t+1) en .npz.

Entrada: dataset/additional_basic_locomotion_2.bvh (23 nodos, 6995 frames @60fps).
Salida: data/locomotion.npz con poses en quaternion (desenrollado), root,
        features de locomocion y split cronologico 80/10/10.

Comentarios en espanol; variables descriptivas en ingles.
"""
import re
import sys
from pathlib import Path

import numpy as np

BVH_PATH = Path(__file__).resolve().parent.parent / "dataset" / "additional_basic_locomotion_2.bvh"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "locomotion.npz"

# Division funcional (indices en orden de aparicion: 0=Hips ... 22=LeftToe).
# Inferior: Hips + piernas (locomocion, soporte, contacto).
INFERIOR_JOINTS = [0, 15, 16, 17, 18, 19, 20, 21, 22]
# Superior: todo lo demas (pecho, brazos, cabeza).
SUPERIOR_JOINTS = [i for i in range(23) if i not in INFERIOR_JOINTS]
ANKLE_LEFT, ANKLE_RIGHT = 21, 17
TOE_LEFT, TOE_RIGHT = 22, 18


def parse_bvh(path):
    """Lee jerarquia y motion. Devuelve (nombres, padres, offsets, orden_canales, datos)."""
    text = Path(path).read_text()
    hierarchy_text, motion_text = text.split("MOTION")
    tokens = re.findall(r"[A-Za-z]+|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", hierarchy_text)
    names, parents, offsets, channel_orders = [], [], [], []
    stack = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("ROOT", "JOINT"):
            name = tokens[i + 1]
            names.append(name)
            parents.append(stack[-1] if stack else -1)
            stack.append(len(names) - 1)
            i += 2
        elif token == "End":
            stack.append(-2)  # marcador: End Site sin datos
            i += 1
        elif token == "OFFSET":
            offsets.append([float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3])])
            i += 4
        elif token == "CHANNELS":
            count = int(tokens[i + 1])
            channel_orders.append(tokens[i + 2:i + 2 + count])
            i += 2 + count
        elif token == "{":
            i += 1
        elif token == "}":
            stack.pop()
            i += 1
        else:
            i += 1
    # Quita offsets de End Sites (van intercalados; nos quedamos con los de nodos reales).
    # Los offsets aparecen en orden de visita: nodo, ..., End Site, ...; filtramos por conteo.
    # Metodo robusto: re-parsea solo bloques ROOT/JOINT.
    names, parents, offsets, channel_orders = parse_hierarchy_robust(hierarchy_text)
    motion_lines = [line for line in motion_text.strip().splitlines() if line.strip()]
    frames = int(motion_lines[0].split(":")[1])
    frame_time = float(motion_lines[1].split(":")[1])
    data = np.array([[float(v) for v in line.split()] for line in motion_lines[2:2 + frames]])
    assert data.shape[0] == frames, f"frames leidos {data.shape[0]} != {frames}"
    return names, parents, offsets, channel_orders, data, frame_time


def parse_hierarchy_robust(hierarchy_text):
    """Parsea solo bloques ROOT/JOINT con su OFFSET y CHANNELS."""
    names, parents, offsets, channel_orders = [], [], [], []
    stack = []          # (indice, profundidad) de nodos con bloque abierto
    depth = 0
    awaiting_block = False
    for line in hierarchy_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("ROOT") or stripped.startswith("JOINT"):
            names.append(stripped.split()[1])
            awaiting_block = True
        elif stripped == "{":
            depth += 1
            if awaiting_block:
                stack.append((len(names) - 1, depth))
                parents.append(stack[-2][0] if len(stack) > 1 else -1)
                offsets.append(None)
                channel_orders.append(None)
                awaiting_block = False
        elif stripped.startswith("OFFSET") and stack and offsets[stack[-1][0]] is None:
            offsets[stack[-1][0]] = [float(v) for v in stripped.split()[1:4]]
        elif stripped.startswith("CHANNELS") and stack:
            parts = stripped.split()
            channel_orders[stack[-1][0]] = parts[2:2 + int(parts[1])]
        elif stripped == "}":
            if stack and stack[-1][1] == depth:
                stack.pop()
            depth -= 1
    return names, parents, offsets, channel_orders


def euler_to_quaternion(euler_degrees, order):
    """Euler (grados, orden BVH p.ej. YXZ) -> quaternion (x, y, z, w)."""
    angles = np.radians(euler_degrees)
    axis_map = {"Xrotation": np.array([1.0, 0.0, 0.0]),
                "Yrotation": np.array([0.0, 1.0, 0.0]),
                "Zrotation": np.array([0.0, 0.0, 1.0])}
    rotation = np.eye(3)
    for axis_name, angle in zip(order, angles):
        axis = axis_map[axis_name]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        cross = np.array([[0.0, -axis[2], axis[1]],
                          [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
        step = (np.eye(3) * cos_a + (1.0 - cos_a) * np.outer(axis, axis) + sin_a * cross)
        rotation = rotation @ step
    return rotation_to_quaternion(rotation)


def rotation_to_quaternion(matrix):
    """Matriz 3x3 -> quaternion (x, y, z, w). Metodo de Shepperd."""
    trace = np.trace(matrix)
    if trace > 0.0:
        root = np.sqrt(trace + 1.0)
        w = 0.5 * root
        root = 0.5 / root
        x = (matrix[2, 1] - matrix[1, 2]) * root
        y = (matrix[0, 2] - matrix[2, 0]) * root
        z = (matrix[1, 0] - matrix[0, 1]) * root
    else:
        i = int(np.argmax([matrix[0, 0], matrix[1, 1], matrix[2, 2]]))
        j = (i + 1) % 3
        k = (i + 2) % 3
        root = np.sqrt(matrix[i, i] - matrix[j, j] - matrix[k, k] + 1.0)
        comp = np.zeros(3)
        comp[i] = 0.5 * root
        root = 0.5 / root
        w = (matrix[k, j] - matrix[j, k]) * root
        comp[j] = (matrix[j, i] + matrix[i, j]) * root
        comp[k] = (matrix[k, i] + matrix[i, k]) * root
        x, y, z = comp
    return np.array([x, y, z, w])


def quaternion_to_matrix(quaternion):
    """Quaternion (x, y, z, w) -> matriz 3x3."""
    x, y, z, w = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_euler_yxz(matrix):
    """Matriz 3x3 -> Euler YXZ en grados (convencion del BVH original)."""
    m = np.asarray(matrix, dtype=np.float64)
    x = np.arcsin(np.clip(-m[1, 2], -1.0, 1.0))
    y = np.arctan2(m[0, 2], m[2, 2])
    z = np.arctan2(m[1, 0], m[1, 1])
    return np.degrees([y, x, z])


def forward_kinematics(quaternions, root_position, offsets, parents):
    """FK: posiciones mundo (nodos, 3) para un frame."""
    count = len(parents)
    global_rotation = [None] * count
    global_position = [None] * count
    for joint in range(count):
        local_rotation = quaternion_to_matrix(quaternions[joint])
        offset = np.array(offsets[joint])
        if parents[joint] == -1:
            global_rotation[joint] = local_rotation
            global_position[joint] = np.array(root_position)
        else:
            parent = parents[joint]
            global_rotation[joint] = global_rotation[parent] @ local_rotation
            global_position[joint] = global_position[parent] + global_rotation[parent] @ offset
    return np.array(global_position)


def main():
    print(f"Leyendo {BVH_PATH} ...")
    names, parents, offsets, channel_orders, data, frame_time = parse_bvh(BVH_PATH)
    joint_count = len(names)
    print(f"Nodos: {joint_count} | Frames: {data.shape[0]} | dt: {frame_time}")
    assert joint_count == 23, f"se esperaban 23 nodos, hay {joint_count}"

    # Columnas por nodo segun sus CHANNELS.
    column = 0
    node_columns = []
    for channels in channel_orders:
        node_columns.append(list(range(column, column + len(channels))))
        column += len(channels)

    root_position = data[:, node_columns[0][:3]]
    quaternions = np.zeros((data.shape[0], joint_count, 4))
    for joint in range(joint_count):
        cols = node_columns[joint]
        channels = channel_orders[joint]
        rotation_cols = [c for c, name in zip(cols, channels) if "rotation" in name]
        rotation_names = [name for name in channels if "rotation" in name]
        for frame in range(data.shape[0]):
            quaternions[frame, joint] = euler_to_quaternion(data[frame, rotation_cols], rotation_names)

    # Desenrollado: evita saltos de signo entre frames consecutivos.
    for joint in range(joint_count):
        for frame in range(1, data.shape[0]):
            if np.dot(quaternions[frame - 1, joint], quaternions[frame, joint]) < 0.0:
                quaternions[frame, joint] *= -1.0

    # FK para alturas de pies (contacto) y velocidad del root.
    print("Calculando FK (alturas de pies) ...")
    foot_heights = np.zeros((data.shape[0], 2))  # [izquierdo, derecho]
    for frame in range(data.shape[0]):
        positions = forward_kinematics(quaternions[frame], root_position[frame], offsets, parents)
        foot_heights[frame, 0] = min(positions[ANKLE_LEFT, 1], positions[TOE_LEFT, 1])
        foot_heights[frame, 1] = min(positions[ANKLE_RIGHT, 1], positions[TOE_RIGHT, 1])
    root_velocity = np.diff(root_position, axis=0, prepend=root_position[:1]) / frame_time

    # Normalizacion: root y features con media/std; quaterniones quedan en [-1, 1].
    root_mean, root_std = root_position.mean(axis=0), root_position.std(axis=0) + 1e-8
    vel_mean, vel_std = root_velocity.mean(axis=0), root_velocity.std(axis=0) + 1e-8
    feet_mean, feet_std = foot_heights.mean(axis=0), foot_heights.std(axis=0) + 1e-8
    root_norm = (root_position - root_mean) / root_std
    vel_norm = (root_velocity - vel_mean) / vel_std
    feet_norm = (foot_heights - feet_mean) / feet_std

    # Limites anatomicos: Euler YXZ por articulacion sobre todo el clip.
    euler_all = np.zeros((data.shape[0], joint_count, 3))
    for frame in range(data.shape[0]):
        for joint in range(joint_count):
            rot = quaternion_to_matrix(quaternions[frame, joint])
            euler_all[frame, joint] = matrix_to_euler_yxz(rot)
    euler_min = euler_all.min(axis=0).astype(np.float32)
    euler_max = euler_all.max(axis=0).astype(np.float32)

    # Pares con HISTORIAL: input[i] usa frames (f-1, f-2, f-3), target[i] es f.
    # Entrada: 92*3 + 3 + 3 + 2 = 284. Salida: quats + root = 95.
    flat_quats = quaternions.reshape(data.shape[0], -1)
    inputs = np.concatenate([flat_quats[2:-1], flat_quats[1:-2], flat_quats[:-3],
                             root_norm[2:-1], vel_norm[2:-1], feet_norm[2:-1]], axis=1)
    targets = np.concatenate([flat_quats[3:], root_norm[3:]], axis=1)

    # Split cronologico 80/10/10.
    total = inputs.shape[0]
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        inputs=inputs.astype(np.float32),
        targets=targets.astype(np.float32),
        train_end=train_end,
        val_end=val_end,
        joint_names=np.array(names),
        inferior_joints=np.array(INFERIOR_JOINTS),
        superior_joints=np.array(SUPERIOR_JOINTS),
        euler_min=euler_min, euler_max=euler_max,
        root_mean=root_mean, root_std=root_std,
        frame_time=frame_time,
    )
    print(f"OK: {OUT_PATH}")
    print(f"inputs {inputs.shape} | targets {targets.shape} | train {train_end} val {val_end - train_end} test {total - val_end}")


if __name__ == "__main__":
    sys.exit(main())
