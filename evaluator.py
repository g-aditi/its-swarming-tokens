import numpy as np
import torch
import torch.nn.functional as F
from utils import entropy_from_logits

class ExactEvaluator:
    def __init__(self, model, tokenizer, token_filter, bootstrap_samples):
        self.model = model
        self.tokenizer = tokenizer
        self.tf = token_filter
        self.bootstrap_samples = bootstrap_samples
        self.device = model.device

    def evaluate(self, token_ids, token_mapping):
        prompts = [token_mapping.build_prompt(self.tf.decode([tid])) for tid in token_ids]

        results = {tid: [] for tid in token_ids}

        for _ in range(self.bootstrap_samples):
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)

            logits = outputs.logits[:, -1, :]
            entropy_batch = entropy_from_logits(logits).float().cpu().numpy().tolist()

            for i, tid in enumerate(token_ids):
                results[tid].append(entropy_batch[i])

        final = {}
        for tid, values in results.items():
            arr = np.array(values, dtype=np.float32)
            final[tid] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "n": len(values),
            }
        return final

    def glitch_check(self, token_ids, token_mapping):
        prompts = [token_mapping.build_prompt(self.tf.decode([tid])) for tid in token_ids]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)

        logits = outputs.logits[:, -1, :]
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1).cpu().tolist()

        return {tid: (preds[i] != tid) for i, tid in enumerate(token_ids)}
