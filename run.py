from hybrid_miner import run_hybrid_glitch_mining
from config import HybridMinerConfig

from hybrid_miner import initialize_model_and_tokenizer

model_path = "/data/datasets/community/huggingface/models--openai--gpt-oss-20b/snapshots/f47b95650b3ce7836072fb6457b362a795993484"

model, tokenizer = initialize_model_and_tokenizer(
    model_path,
    device="auto"
)

config = HybridMinerConfig(
    population_size=24,
    generations=5,
    k_neighbors=64,
    batch_eval_size=4,
    bootstrap_samples=2,
)

results = run_hybrid_glitch_mining(model, tokenizer, config)

print("\n=== Glitch Tokens Found ===\n")
for r in results:
    print(f"ID: {r['id']}")
    print(f"Token: {repr(r['token'])}")
    print(f"Entropy mean: {r['entropy_mean']}")
    print("-" * 40)
