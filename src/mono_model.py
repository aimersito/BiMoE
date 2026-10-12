"""BiMoE-1X: UNA sola red que hace todo (trial de viabilidad).

Misma data, mismas losses, mismo regimen y mismas puertas que BiMoE-10K v5.
Unica variable del experimento: arquitectura (MLP unico vs router+2 expertos).
Si esto falla igual, la arquitectura queda descartada; si pasa, la complejidad
era el problema. Cualquiera de los dos resultados informa.

Salida: 95 numeros = 92 deltas de quats + 3 deltas de root (residual, como v4/v5).
"""
import torch
import torch.nn as nn

INPUT_SIZE = 285  # 92 quats x3 + 3 root + 3 vel + 2 pies + 1 fase
HIDDEN = 56       # 56^2 + 382*56 + 95 = 24.623 params (~25k, sobrado)
OUTPUT_SIZE = 95  # 92 delta-quats + 3 delta-root
N_JOINTS = 23


class MonoNet(nn.Module):
    """MLP unico: estado -> delta completo. Sin router, sin expertos."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, OUTPUT_SIZE),
        )

    def forward(self, state):
        return self.net(state)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def predict_next(delta95, base_quats, base_root_raw, root_std):
    """delta95 (B,95) -> quats unitarios (B,23,4) + root en metros (B,3).
    Residual: siguiente = base + delta, quats renormalizados."""
    dq = delta95[:, :92].reshape(-1, N_JOINTS, 4)
    full = base_quats + dq
    norm = full.norm(dim=-1, keepdim=True).clamp_min(1e-4)
    full = full / norm
    root_raw = base_root_raw + delta95[:, 92:95] * root_std
    return full, root_raw
