from dataclasses import dataclass

@dataclass
class HybridMinerConfig:
    population_size: int = 64
    generations: int = 30
    k_neighbors: int = 128
    batch_eval_size: int = 8
    bootstrap_samples: int = 8
    ucb_beta: float = 1.0
    prior_lambda: float = 0.0
    seed: int = 42
    device: str = None
