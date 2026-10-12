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


class MonoFiLM(nn.Module):
    """Mono + fase que MODULA la red (truco PFNN barato, sin 4x pesos).

    La fase (ultimo float del estado, 0-1) genera gamma/beta por capa oculta:
    h = ReLU(gamma * (W h + b) + beta). Init gamma=1/beta=0 = arranca como
    MonoNet plana; aprende la modulacion solo si tiene senal.
    Mismo input (285) y output (95) que MonoNet; predict_next intacto.
    Vaso NUEVO: no admite --resume de pesos mono (formas distintas)."""

    FILM_HIDDEN = 32

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(INPUT_SIZE, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, OUTPUT_SIZE)
        self.relu = nn.ReLU()
        self.film = nn.Sequential(
            nn.Linear(2, self.FILM_HIDDEN), nn.ReLU(),
            nn.Linear(self.FILM_HIDDEN, 4 * hidden),
        )
        # Init neutro: gamma=1, beta=0 (empieza como MLP plana).
        with torch.no_grad():
            self.film[-1].weight.zero_()
            self.film[-1].bias.zero_()
            self.film[-1].bias[:hidden] = 0.0      # beta1
            self.film[-1].bias[hidden:2 * hidden] = 1.0  # gamma1
            self.film[-1].bias[2 * hidden:3 * hidden] = 0.0  # beta2
            self.film[-1].bias[3 * hidden:] = 1.0  # gamma2

    def forward(self, state):
        phase = state[:, -1:]
        tau = phase * (2.0 * torch.pi)
        emb = torch.cat([torch.sin(tau), torch.cos(tau)], dim=-1)
        film = self.film(emb)
        h = self.hidden
        beta1, gamma1, beta2, gamma2 = film.split(h, dim=-1)
        out = self.relu(gamma1 * self.fc1(state) + beta1)
        out = self.relu(gamma2 * self.fc2(out) + beta2)
        return self.fc3(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class RootNet(nn.Module):
    """Cabeza root aislada (apuesta 2 del analisis): MLP 9->32->32->3.

    Input: root_n(3)+vel_n(3)+feet_n(2)+fase(1) = 9 dims del estado.
    Output: delta-root en unidades normalizadas (x root_std en predict).
    Init ultima capa a 0 = arranca como velocidad-cero (seguro).
    El trunk nunca recibe gradiente de root: la loss root solo toca esta red."""

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, state):
        root_n = state[:, 276:279]
        vel_n = state[:, 279:282]
        feet_n = state[:, 282:284]
        phase = state[:, -1:]
        x = torch.cat([root_n, vel_n, feet_n, phase], dim=-1)
        return self.net(x)

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
