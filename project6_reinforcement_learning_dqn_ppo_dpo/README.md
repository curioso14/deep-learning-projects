# Reinforcement Learning: DQN, PPO & DPO Fine-tuning

[#reinforcement-learning-dqn-ppo--dpo-fine-tuning](#reinforcement-learning-dqn-ppo--dpo-fine-tuning)

Implementation of value-based and policy-gradient RL methods on a classic control environment, plus preference-based fine-tuning of a language model using Direct Preference Optimization.

**Stack:** Python · PyTorch · Gymnasium · HuggingFace Transformers · PEFT (LoRA) · NumPy · Matplotlib

---

## Notebooks

[#notebooks](#notebooks)

| Notebook | Method | Environment / Model |
|---|---|---|
| `hw3-dqn-agent.ipynb` | Deep Q-Network (DQN) | LunarLander-v3 |
| `hw3-ppo-agent.ipynb` | Proximal Policy Optimization (PPO) | LunarLander-v3 |
| `hw3-dpo.ipynb` | Direct Preference Optimization (DPO) with LoRA | Qwen2.5-0.5B |

---

## Skills

[#skills](#skills)

- **Value-based RL:** DQN with experience replay buffer, target network, epsilon-greedy exploration with decay
- **Policy-gradient RL:** PPO with clipped surrogate objective, advantage estimation, on-policy rollout collection
- **Preference-based fine-tuning:** DPO loss formulation, LoRA-based parameter-efficient fine-tuning (rank/alpha tuning) on a pretrained LLM
- **Training pipeline design:** reward curve tracking, hyperparameter tuning (learning rate, batch size, replay buffer size, DPO beta), convergence analysis

---

## Setup

[#setup](#setup)
