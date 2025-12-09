import torch
import torch.nn.functional as F
import numpy as np

class TokenMapping:
    def __init__(self, model, token_filter, prior_lambda=0.0):
        self.model = model
        self.tf = token_filter
        self.embedding = model.lm_head.weight.detach()
        self.vocab_size, self.emb_dim = self.embedding.shape

        self.skip_tokens = set(self.tf.filter_token())
        self.allowed_ids = [i for i in range(self.vocab_size) if i not in self.skip_tokens]

        self.allowed_ids_tensor = torch.tensor(self.allowed_ids, device=self.embedding.device)
        self.allowed_embeddings = self.embedding[self.allowed_ids_tensor]
        self.allowed_norms = F.normalize(self.allowed_embeddings, p=2, dim=1)

        self.prior_lambda = prior_lambda
        self.token_priors = self._build_token_prior() if prior_lambda > 0 else None

    def _build_token_prior(self):
        scores = []
        for tid in self.allowed_ids:
            s = self.tf.decode([tid])
            score = 0.0
            if all(ord(c) < 128 for c in s):
                score += 1.0
            if any(c.isalnum() for c in s):
                score += 0.5
            scores.append(score)

        arr = np.array(scores, dtype=np.float32)
        arr = arr / arr.sum() if arr.sum() != 0 else np.ones_like(arr) / len(arr)

        return torch.tensor(arr, device=self.embedding.device)

    def embedding_to_token(self, pos: torch.Tensor) -> int:
        pos_norm = F.normalize(pos.unsqueeze(0), p=2, dim=1)
        sims = (self.allowed_norms @ pos_norm.squeeze(0)).cpu()
        idx = torch.argmax(sims).item()
        return self.allowed_ids[idx]

    def nearest_neighbors(self, token_id: int, k: int):
        ref = F.normalize(self.embedding[token_id].unsqueeze(0), p=2, dim=1)
        sims = (self.allowed_norms @ ref.squeeze(0)).cpu()
        sorted_idx = torch.topk(sims, k=min(k, len(self.allowed_ids))).indices.tolist()
        ids = [self.allowed_ids[i] for i in sorted_idx if self.allowed_ids[i] != token_id]
        return ids[:k]
