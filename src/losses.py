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
    movement = torch.diff(feet[:, :, [0, 2]], dim=0).pow(2).sum(dim=-1).sqrt()
    contact_pairs = contact[:-1] & contact[1:]
    if contact_pairs.sum() == 0:
        return movement.mean() * 0.0
    return (movement * contact_pairs.float()).sum() / contact_pairs.sum()


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
    ex = torch.asin(torch.clamp(-m12, -1.0, 1.0))
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


def assemble_prediction(pred_lower, pred_upper, inferior_idx, superior_idx,
                        root_mean, root_std, n_joints=23):
    """Junta expertos -> quats unitarios (B,J,4) + root en metros (B,3)."""
    batch = pred_lower.shape[0]
    pred_quats = torch.zeros(batch, n_joints, 4, device=pred_lower.device,
                             dtype=pred_lower.dtype)
    pred_quats[:, inferior_idx] = pred_lower[:, :36].reshape(batch, 9, 4)
    pred_quats[:, superior_idx] = pred_upper.reshape(batch, 14, 4)
    pred_quats = functional.normalize(pred_quats, dim=-1)
    pred_root = pred_lower[:, 36:39] * root_std + root_mean  # a metros para FK
    return pred_quats, pred_root


def total_loss(pred_lower, pred_upper, target, offsets, parents, inferior_idx, superior_idx,
               root_mean, root_std, euler_min, euler_max, ankle_idx=(17, 21), toe_idx=(18, 22),
               weight_pose=1.0, weight_foot=0.5, weight_smooth=0.3, weight_limit=0.5,
               weight_reg=1e-5, model=None):
    """Junta todo. target: (batch, 95) = 92 quats + 3 root."""
    pred_quats, pred_root = assemble_prediction(pred_lower, pred_upper, inferior_idx,
                                               superior_idx, root_mean, root_std)
    target_quats = target[:, :92].reshape(-1, 23, 4)
    pose = quaternion_sign_invariant_loss(pred_quats, target_quats)
    foot = foot_sliding_loss(pred_quats, pred_root, offsets, parents, ankle_idx, toe_idx)
    smooth = smoothness_loss(pred_quats)
    limit = joint_limit_loss(pred_quats, euler_min, euler_max)
    reg = sum(p.pow(2).sum() for p in model.parameters()) if model is not None else 0.0
    total = (weight_pose * pose + weight_foot * foot + weight_smooth * smooth
             + weight_limit * limit + weight_reg * reg)
    return total, {"pose": pose.detach(), "foot": foot.detach(),
                   "smooth": smooth.detach(), "limit": limit.detach()}
