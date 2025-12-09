from config import HybridMinerConfig
from surrogate import SurrogateModel
from evaluator import ExactEvaluator
from token_mapping import TokenMapping
from search import HybridSearch
from utils import set_seed
from tokenfilter import TokenFilter
from llm_template import get_template_for_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def initialize_model_and_tokenizer(model_path, device=None):
    # auto device
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Always load GPT-OSS-20B in BF16
    dtype = torch.bfloat16
    print(f"Loading model with dtype={dtype} ...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,     
        low_cpu_mem_usage=True,
        device_map=None
    )

    print(f"Moving model to {device} ...")
    model = model.to(device)

    model.eval()
    model.requires_grad_(False)
    return model, tokenizer

def HybridGlitchMiner(model, tokenizer, config: HybridMinerConfig):
    return run_hybrid_glitch_mining(model, tokenizer, config)

def run_hybrid_glitch_mining(model, tokenizer, config: HybridMinerConfig):
    set_seed(config.seed)

    tf = TokenFilter(model, tokenizer)
    tm = TokenMapping(model, tf, prior_lambda=config.prior_lambda)

    tm.chat_template = get_template_for_model(model.config._name_or_path)
    tm.build_prompt = lambda token: (
        (tm.chat_template.system_format.format(content="") if tm.chat_template.system_format else "") +
        tm.chat_template.user_format.format(content=f'Please repeat the string: "«{token}»"') +
        ' Sure, the string is: "«'
    )

    surrogate = SurrogateModel(model, tokenizer, tf, config.k_neighbors)
    evaluator = ExactEvaluator(model, tokenizer, tf, config.bootstrap_samples)

    search = HybridSearch(config, surrogate, evaluator, tm)
    
    glitches, stats = search.run()

    results = []
    for tid in glitches:
        tok = tokenizer.decode([tid])
        ent_info = stats.get(tid, {})
        results.append({
            "id": tid,
            "token": tok,
            "entropy_mean": ent_info.get("mean"),
            "entropy_std": ent_info.get("std"),
            "n_samples": ent_info.get("n"),
        })

    return results
