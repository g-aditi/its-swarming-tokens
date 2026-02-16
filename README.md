# It’s Swarming Tokens  
**Hybrid GA + PSO for Glitch Token Discovery in LLMs**

This project implements a hybrid evolutionary search algorithm for discovering **glitch tokens** in large language models (LLMs).

A *glitch token* is a token that a model fails to reproduce when explicitly prompted to repeat it, revealing instability in embedding geometry, decoding behavior, or routing (especially in mixture-of-experts models).

The method extends gradient-based approaches (e.g., *GlitchMiner*) by combining:

- **Genetic Algorithm (GA)** over discrete token IDs  
- **Particle Swarm Optimization (PSO)** in embedding space  
- **Entropy-gradient surrogate model** for fast candidate ranking  
- **Bootstrap + UCB scoring** for variance-aware evaluation  

This enables more global exploration of the token embedding landscape and improved glitch discovery under fixed evaluation budgets.

---

## Core Idea

Glitch-token mining is treated as a **mixed discrete–continuous optimization problem**:

- Discrete search over vocabulary tokens  
- Continuous search over embedding space  

Gradient-only local search can stall in irregular regions (particularly in MoE architectures). The hybrid GA+PSO approach enables non-local exploration and better coverage.

---

## Summary

Tested on:

- Llama-3.1-8B-Instruct  
- Mistral-7B-Instruct-v0.3  
- GPT-OSS-20B  

The hybrid method consistently discovers more glitch tokens than gradient-based baselines under equal evaluation budgets, with especially strong gains on mixture-of-experts models.

---

## Why This Matters

Glitch tokens expose:

- Embedding-space degeneracies  
- Model instability  
- Potential reliability and safety vulnerabilities  

Systematic detection helps probe structural weaknesses in modern LLMs.

---

## Status

Research prototype. Built to explore algorithmic extensions to glitch-token mining and LLM reliability analysis.
