import torch
import torch.nn.functional as F
from utils import entropy_from_logits

class SurrogateModel:
    def __init__(self, model, tokenizer, token_filter, k_neighbors):
        self.model = model
        self.tokenizer = tokenizer
        self.tf = token_filter
        self.k = k_neighbors
        self.device = model.device

        self.embedding = model.lm_head.weight.detach()

    def surrogate(self, reference_token_id: int, token_mapping):
        tok_text = self.tf.decode([reference_token_id])
        chat_template = token_mapping.chat_template
        formatted_prompt = token_mapping.build_prompt(tok_text)

        enc = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]

        # find position of inserted token after quote
        quote_id = self.tokenizer.encode('"«', add_special_tokens=False)[-1]
        try:
            token_pos = (input_ids[0] == quote_id).nonzero(as_tuple=True)[0].item() + 1
        except:
            token_pos = input_ids.shape[1] - 1

        inputs_embeds = self.model.get_input_embeddings()(input_ids).detach()
        inputs_embeds.requires_grad_(True)

        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
        logits = outputs.logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)
        entropy = entropy_from_logits(logits)

        grads = torch.autograd.grad(entropy, inputs_embeds, retain_graph=False)[0]
        grad_vec = grads[0, token_pos, :].detach()

        # nearest neighbors
        neighbor_ids = token_mapping.nearest_neighbors(reference_token_id, self.k)
        ref_emb = self.embedding[reference_token_id]

        neighbor_embs = self.embedding[torch.tensor(neighbor_ids, device=ref_emb.device)]
        delta = neighbor_embs - ref_emb.unsqueeze(0)

        delta_dot_grad = (delta @ grad_vec).to(torch.float32).cpu().numpy()
        lin_scores = entropy.item() + delta_dot_grad


        return neighbor_ids, lin_scores, entropy.item()
