"""Shared local judge model loading for QIMMA generative tasks."""

import os
from functools import lru_cache


DEFAULT_JUDGE_MODEL = "Qwen/Qwen3.5-27B"


@lru_cache(maxsize=1)
def load_judge():
    """Load one judge model/tokenizer pair for all QIMMA tasks in this process."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ModuleNotFoundError(
            "QIMMA local judges require transformers and accelerate."
        ) from exc

    model_name = os.getenv("QIMMA_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    device_map = os.getenv("QIMMA_JUDGE_DEVICE_MAP", "auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map=device_map
    )
    model.eval()
    return model, tokenizer
