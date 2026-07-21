# Reinforcement Learning: DQN, PPO & DPO Fine-tuning

[#reinforcement-learning-dqn-ppo--dpo-fine-tuning](#reinforcement-learning-dqn-ppo--dpo-fine-tuning)

Implementation of value-based and policy-gradient RL methods on LunarLander-v3, plus preference-based fine-tuning of a language model using Direct Preference Optimization.

**Stack:** Python · PyTorch · Gymnasium (Box2D) · HuggingFace Transformers · TRL · PEFT (LoRA) · NumPy · Matplotlib

---

## Notebooks

[#notebooks](#notebooks)

| Notebook | Method | Environment / Model |
|---|---|---|
| `hw3-dqn-agent.ipynb` | Deep Q-Network (DQN) | LunarLander-v3 |
| `hw3-ppo-agent.ipynb` | Proximal Policy Optimization (PPO) | LunarLander-v3 |
| `hw3-dpo.ipynb` | Direct Preference Optimization (DPO) with LoRA | Qwen2.5-0.5B-Instruct |

---

## 1. DQN — LunarLander-v3

[#1-dqn--lunarlander-v3](#1-dqn--lunarlander-v3)

MLP Q-network (2 hidden layers, 128 units) with experience replay (buffer size 50,000) and a periodically-synced target network.

**Key hyperparameters:** γ=0.99 · ε decay 1.0→0.01 (×0.997/episode) · lr=5e-3 · batch size 128 · target update every 5 episodes

**Results (10 eval episodes, greedy policy):** 5/10 landings successful (reward ≥ 200) · mean reward 70.1 · max 297.1 · min -184.1

**Ablation study** (400 episodes each, isolating one hyperparameter):
- `epsilon_decay`: 0.990 → **65.4** · 0.995 → -96.0 · 0.999 → 14.5
- `batch_size`: 32 → -31.9 · 64 → -98.7 · **128 → 16.4**
- `target_update_freq`: **5 → -197.8** · 10 → -61.6 · 20 → -89.0 (all underperformed the main run's freq=5, suggesting high variance at short ablation horizons)

---

## 2. PPO — LunarLander-v3

[#2-ppo--lunarlander-v3](#2-ppo--lunarlander-v3)

Actor-Critic with clipped surrogate objective and GAE, trained on 4 parallel vectorized environments over 2M timesteps.

**Key hyperparameters:** γ=0.99 · GAE λ=0.95 · clip ε=0.2 · lr=2.5e-4 · 8 epochs/update · batch size 128 · entropy coef 0.02 · value coef 0.5 · max grad norm 0.5

**Results (10 eval episodes, deterministic policy):** 5/10 landings successful · mean reward 150.6 · max 287.7 · min -42.2

**Ablation study** (500k timesteps each):
- `learning_rate`: 1e-4 → -18.9 · 3e-4 → -119.6 · **5e-4 → 79.1**
- `n_epochs`: 4 → 26.3 · **8 → 43.4** · 16 → 35.5
- `ent_coef`: **0.005 → 79.0** · 0.010 → 1.8 · 0.020 → 38.5

---

## 3. DPO Fine-tuning — Qwen2.5-0.5B-Instruct

[#3-dpo-fine-tuning--qwen25-05b-instruct](#3-dpo-fine-tuning--qwen25-05b-instruct)

Fine-tuned the base model with DPO to prefer concise responses over verbose ones, using LoRA for parameter-efficient training.

**Dataset:** 453 preference pairs (chosen = concise, rejected = verbose)

**LoRA config:** r=16 · alpha=32 · dropout=0.05 · target modules: `q_proj, v_proj, k_proj, o_proj`

**DPO config:** β=0.1 · lr=5e-5 · 2 epochs · batch size 2 (×4 grad accumulation) · max length 1024

**Results:**
| | Loss | Avg. response length |
|---|---|---|
| Start of training | 0.6194 | — |
| End of training | 0.2324 | — |
| Before fine-tuning | — | 177.0 words |
| After fine-tuning | — | 21.0 words |

**Average response length reduction: 88.1%**, with qualitatively preserved answer content across all 3 held-out test prompts.

---

## Skills

[#skills](#skills)

- **Value-based RL:** DQN with experience replay, target network stabilization, epsilon-greedy exploration with decay
- **Policy-gradient RL:** PPO with clipped surrogate objective, GAE, vectorized environment rollout collection, entropy regularization
- **Preference-based fine-tuning:** DPO loss formulation, LoRA-based parameter-efficient fine-tuning on a pretrained LLM via TRL's `DPOTrainer`
- **Hyperparameter analysis:** systematic ablation studies isolating individual hyperparameters across all three methods
- **Training pipeline design:** reward curve tracking, moving-average smoothing with std bands, convergence analysis

---

## Setup

[#setup](#setup)

pip install torch "gymnasium[box2d]" numpy matplotlib
pip install trl[peft] peft transformers datasets torchao
