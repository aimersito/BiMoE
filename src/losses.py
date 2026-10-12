"""Perdidas BiMoE-10K: fidelidad (invariante al signo) + suavidad + contacto.

El quaternion q y -q representan la misma rotacion: la fidelidad mide contra
el mas cercano. El foot sliding se penaliza con FK en torch.
"""
import torch
import torch.nn.functional as functional


def quaternion_sign_invariant_loss(predicted, target):
    """MSE contra el mas cercano entre target y -target, por articulacion."""
    positive = (predicted - target) ** 2
    negative = (predicted + target) ** 2
    per_joint = torch.minimum(positive.sum(dim=-1), negative.sum(dim=-1))
    return per_joint.mean()


def quaternion_phuber_loss(predicted, target, eps):
    """Sign-invariante con pseudo-Huber elementwise (robusta a outliers)."""
    pos = torch.sqrt((predicted - target).pow(2) + eps * eps) - eps
    neg = torch.sqrt((predicted + target).pow(2) + eps * eps) - eps
    per_joint = torch.minimum(pos.sum(dim=-1), neg.sum(dim=-1))
    return per_joint.mean()


def forward_kinematics_batch(quaternions, root_position, offsets, parents):
    """FK por batch: (batch, nodos, 4) + (batch, 3) -> (batch, nodos, 3)."""
    batch, joint_count, _ = quaternions.shape
    global_rot = [None] * joint_count
    global_pos = [None] * joint_count
    for joint in range(joint_count):
        x, y, z, w = quaternions[:, joint].unbind(dim=-1)
        local = torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ], dim=-1).reshape(batch, 3, 3)
        offset = offsets[joint].expand(batch, 3)
        if parents[joint] == -1:
            global_rot[joint] = local
            global_pos[joint] = root_position
        else:
            parent = parents[joint]
            global_rot[joint] = global_rot[parent] @ local
            global_pos[joint] = global_pos[parent] + (global_rot[parent] @ offset.unsqueeze(-1)).squeeze(-1)
    return torch.stack(global_pos, dim=1)


def foot_sliding_loss(pred_quats, pred_root, offsets, parents, ankle_idx, toe_idx, threshold=0.05):
    """Distancia horizontal de los pies que deberian estar fijos entre frames."""
    positions = forward_kinematics_batch(pred_quats, pred_root, offsets, parents)
    feet = torch.stack([
        torch.minimum(positions[:, ankle_idx[0]], positions[:, toe_idx[0]]),
        torch.minimum(positions[:, ankle_idx[1]], positions[:, toe_idx[1]]),
    ], dim=1)  # (batch, 2, 3)
    heights = feet[:, :, 1]
    contact = heights < (heights.min() + threshold)
    # eps dentro del sqrt: con salidas (casi) constantes los diffs son EXACTOS
    # a 0 y el backward de sqrt da 0/0=NaN (forward finito). Escala intacta.
    movement = torch.sqrt(torch.diff(feet[:, :, [0, 2]], dim=0).pow(2).sum(dim=-1) + 1e-8)
    contact_pairs = contact[:-1] & contact[1:]
    if contact_pairs.sum() == 0:
        return movement.mean() * 0.0
    return (movement * contact_pairs.float()).sum() / contact_pairs.sum()


def foot_contact_loss(pred_feet_seq, contact_mask, shape="mse", eps=0.005):
    """v5: pies ETIQUETADOS en contacto deben quedarse quietos.
    pred_feet_seq: (B,K,2,3) con gradiente. contact_mask: (B,K-1,2) bool
    (etiqueta del frame k: altura<0.10m AND velocidad<0.15m/s, estándar Kimodo).
    Castiga desplazamiento horizontal del pie en contacto (HuMoR/MDM).
    shape: 'mse' (default intacto, cuadrado), 'sqrt' (L2 sin cuadrar estilo
    codigo MoDi: gradiente 55-200x mayor con d=0.0025-0.009 m/f), 'phuber'."""
    disp = torch.diff(pred_feet_seq[:, :, :, [0, 2]], dim=1).norm(dim=-1)
    if shape == "sqrt":
        movement = disp
    elif shape == "phuber":
        movement = torch.sqrt(disp.pow(2) + eps * eps) - eps
    else:
        movement = disp.pow(2)
    if contact_mask.sum() == 0:
        return movement.mean() * 0.0
    return (movement * contact_mask.float()).sum() / contact_mask.float().sum()


def pseudo_huber_mean(err, eps):
    """Media pseudo-Huber sqrt(e^2+eps^2)-eps: lineal en grande, cuadratica en chico.
    Mismo cero y escala que MSE cerca de 0 (dividido por eps para emparejar)."""
    return (torch.sqrt(err.pow(2) + eps * eps) - eps).mean()


def smoothness_loss(pred_quats):
    """Penaliza cambios bruscos de velocidad angular entre frames."""
    velocity = torch.diff(pred_quats, dim=0)
    accel = torch.diff(velocity, dim=0)
    return accel.pow(2).mean()


def quaternion_to_euler_yxz(quaternions):
    """(..., 4) quats -> (..., 3) Euler YXZ en grados (version torch)."""
    x, y, z, w = quaternions.unbind(dim=-1)
    m00 = 1 - 2 * (y * y + z * z)
    m02 = 2 * (x * z + y * w)
    m10 = 2 * (x * y + z * w)
    m11 = 1 - 2 * (x * x + z * z)
    m12 = 2 * (y * z - x * w)
    m22 = 1 - 2 * (x * x + y * y)
    ex = torch.asin(torch.clamp(-m12, -1.0 + 1e-6, 1.0 - 1e-6))
    ey = torch.atan2(m02, m22)
    ez = torch.atan2(m10, m11)
    return torch.stack([ey, ex, ez], dim=-1) * (180.0 / 3.141592653589793)


def joint_limit_loss(pred_quats, euler_min, euler_max, margin=5.0):
    """Castiga angulos fuera del rango anatomico visto en el BVH (+margen).
    Todo en unidades de 180 grados para que la escala combine con pose."""
    euler = quaternion_to_euler_yxz(pred_quats) / 180.0
    low = (euler_min - margin) / 180.0
    high = (euler_max + margin) / 180.0
    below = torch.relu(low - euler)
    above = torch.relu(euler - high)
    return (below.pow(2) + above.pow(2)).mean()


def assemble_prediction(pred_lower, pred_upper, base_quats, base_root_raw,
                        inferior_idx, superior_idx, root_mean, root_std, n_joints=23):
    """Junta expertos (RESIDUALES) -> quats unitarios (B,J,4) + root en metros (B,3).
    v4: el modelo predice DELTAS, no pose absoluta (rompe el atractor de punto fijo).
      full = norm(base + delta_quats); root = base_root + delta_root*root_std."""
    batch = pred_lower.shape[0]
    delta_quats = torch.zeros(batch, n_joints, 4, device=pred_lower.device,
                              dtype=pred_lower.dtype)
    delta_quats[:, inferior_idx] = pred_lower[:, :36].reshape(batch, 9, 4)
    delta_quats[:, superior_idx] = pred_upper.reshape(batch, 14, 4)
    # Residual: siguiente = base + delta, renormalizado a unitario.
    pred_quats = base_quats + delta_quats
    # Norma con piso: un vector cero exacto daria NaN al normalizar.
    # Piso 1e-4 (no 1e-8): el backward escala con 1/norma y con 1e-8 un
    # gradiente inf entra a clip_grad_norm_ (inf*0=NaN) y envenena los pesos.
    norm = pred_quats.norm(dim=-1, keepdim=True).clamp_min(1e-4)
    pred_quats = pred_quats / norm
    # Delta de root en unidades normalizadas (escala ~unitaria como el resto).
    pred_root = base_root_raw + pred_lower[:, 36:39] * root_std  # a metros para FK
    return pred_quats, pred_root, delta_quats


def total_loss(pred_lower, pred_upper, base, target, offsets, parents, inferior_idx, superior_idx,
               root_mean, root_std, euler_min, euler_max, ankle_idx=(17, 21), toe_idx=(18, 22),
               weight_pose=1.0, weight_foot=0.5, weight_smooth=0.3, weight_limit=0.15,
               weight_root=1.0, weight_reg=1e-5, model=None):
    """Junta todo. base/target: (batch, 95) = 92 quats + 3 root_norm.
    v4: salidas residuales; base = estado previo (quats + root_norm).
    v5: limite 0.5->0.15 (a 16k params el auxiliar ahogaba a pose: era 29%)."""
    base_quats = base[:, :92].reshape(-1, 23, 4)
    base_root_raw = base[:, 92:95] * root_std + root_mean
    pred_quats, pred_root, _ = assemble_prediction(pred_lower, pred_upper, base_quats,
                                                   base_root_raw, inferior_idx,
                                                   superior_idx, root_mean, root_std)
    target_quats = target[:, :92].reshape(-1, 23, 4)
    true_root_raw = target[:, 92:95] * root_std + root_mean
    pose = quaternion_sign_invariant_loss(pred_quats, target_quats)
    # Root supervisado directo (antes solo aprendia via FK del foot loss).
    root = (((pred_root - true_root_raw) / root_std).pow(2)).mean()
    foot = foot_sliding_loss(pred_quats, pred_root, offsets, parents, ankle_idx, toe_idx)
    smooth = smoothness_loss(pred_quats)
    limit = joint_limit_loss(pred_quats, euler_min, euler_max)
    reg = sum(p.pow(2).sum() for p in model.parameters()) if model is not None else 0.0
    total = (weight_pose * pose + weight_root * root + weight_foot * foot
             + weight_smooth * smooth + weight_limit * limit + weight_reg * reg)
    return total, {"pose": pose.detach(), "root": root.detach(), "foot": foot.detach(),
                   "smooth": smooth.detach(), "limit": limit.detach()}
