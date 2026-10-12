# BiMoE-10K

Reemplazo de Motion Matching con red neuronal: **router + 2 expertos**.
Recibe el estado del jugador y devuelve la siguiente pose, un frame por vez.

## Estado

Prueba con 1 clip de marcha (`additional_basic_locomotion_2.bvh`, 23 nodos,
6995 frames @60fps). Modelo de 16.403 params, entrada con historial de 3 frames.

## Estructura

- `src/preprocess.py` — BVH → pares en `data/locomotion.npz`
- `src/bimoe_model.py` — router + 2 expertos (salidas disjuntas concatenadas)
- `src/losses.py` — fidelidad invariante al signo + foot sliding (FK) +
  suavidad + límites anatómicos
- `src/train.py` — entreno con ruido gaussiano (robustez)
- `src/infer.py` — rollout + exporta BVH
- `src/play.py` — demo jugable GLFW/OpenGL (WASD)
- `src/render.py`, `src/viewer.py` — video y visor Open3D
- `notebooks/bimoe10k_kaggle.ipynb` — entreno en Kaggle T4

## Uso local

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python src/preprocess.py
.venv/bin/python src/play.py   # WASD mueve, Espacio frena, Esc sale
```

## Kaggle

```bash
git clone https://github.com/aimersito/BiMoE.git
cd BiMoE && python3 src/train.py --data data/locomotion.npz --epochs 200
```
