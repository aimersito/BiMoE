"""BiMoE-10K: nucleo instanciado para locomocion (23 nodos, 1 clip caminar).

Arquitectura: router (contexto) + 2 expertos con salidas disjuntas concatenadas.
Total ~10.435 parametros. Comentarios en espanol; variables en ingles.
"""
import torch
import torch.nn as nn

INPUT_SIZE = 284  # 92 quats x3 (t, t-1, t-2) + 3 root + 3 vel + 2 pies
CONTEXT_SIZE = 8
INFERIOR_OUT = 39  # 9 articulaciones x 4 + 3 root
SUPERIOR_OUT = 56  # 14 articulaciones x 4


class Router(nn.Module):
    """Mira el estado y produce el contexto que coordina a los expertos."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, 16), nn.ReLU(),
            nn.Linear(16, 12), nn.ReLU(),
            nn.Linear(12, CONTEXT_SIZE),
        )

    def forward(self, state):
        return self.network(state)


class Expert(nn.Module):
    """Experto de una region funcional: entrada + contexto -> su parte del cuerpo."""

    def __init__(self, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE + CONTEXT_SIZE, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, state, context):
        return self.network(torch.cat([state, context], dim=-1))


class BiMoE(nn.Module):
    """Nucleo instanciado: compone por concatenacion (salidas disjuntas)."""

    def __init__(self):
        super().__init__()
        self.router = Router()
        self.lower_expert = Expert(INFERIOR_OUT)
        self.upper_expert = Expert(SUPERIOR_OUT)

    def forward(self, state):
        context = self.router(state)
        lower = self.lower_expert(state, context)
        upper = self.upper_expert(state, context)
        return lower, upper, context

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
