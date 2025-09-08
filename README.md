# MoE + RL on GSM8K (Single-GPU)

Fine-tune an **off-the-shelf MoE** model with **policy-gradient RL (PPO)** on **GSM8K**.

- Default: `microsoft/Phi-tiny-MoE-instruct` (≈3.8B total / 1.1B activated).
- Single-GPU friendly via **LoRA** + optional **4‑bit** loading.
- Clean modular code: model / data / RL / training / eval.

See `configs/default.yaml` and run:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
```

Push to GitHub:

```bash
git init && git add . && git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
