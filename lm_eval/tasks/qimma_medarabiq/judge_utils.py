"""LLM-as-judge scoring for the generative QIMMA MedArabiQ subset."""

import os
import re
from functools import lru_cache
from typing import Any

from lm_eval.tasks._qimma_reasoning import answer_after_thinking
from lm_eval.tasks.qimma_medarabiq.qa_utils import doc_to_target


DEFAULT_JUDGE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_MAX_NEW_TOKENS = 512
_VERDICT_RE = re.compile(r"\[\[\s*([01])\s*\]\]")

_SYSTEM_PROMPT = """أنت محكّم خبير في الأسئلة الطبية العربية. قيّم إجابة النموذج اعتماداً فقط على السؤال والإجابة المرجعية.

قواعد التقييم:
- امنح 1 إذا كانت الإجابة صحيحة في المعنى وتغطي الجواب المطلوب، حتى مع اختلاف الصياغة.
- امنح 0 إذا كانت خاطئة، أو تناقض المرجع، أو تفتقد معلومة جوهرية، أو لا تجيب عن السؤال.
- لا تعاقب على اختلافات أسلوبية أو تفاصيل إضافية صحيحة.
- إذا احتوت الإجابة على معلومات إضافية خاطئة أو متناقضة، امنح 0.

المطلوب منك هو التقييم فقط، ولا حاجة إلى أي تفكير أو تعليل أو شرح أو نص إضافي.
يجب أن يكون ردك كاملاً واحداً من الخيارين التاليين فقط: [[1]] أو [[0]]."""


def _build_judge_prompt(question: str, reference: str, prediction: str) -> str:
    return (
        f"السؤال:\n{question.strip()}\n\n"
        f"الإجابة المرجعية:\n{reference.strip()}\n\n"
        f"إجابة النموذج المراد تقييمها:\n{prediction.strip()}\n\n"
        "الحكم:"
    )


def _parse_verdict(text: str) -> float:
    matches = _VERDICT_RE.findall(text)
    if not matches or len(set(matches)) != 1:
        return 0.0
    return float(matches[-1])


@lru_cache(maxsize=1)
def _load_judge():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ModuleNotFoundError(
            "qimma_medarabiq requires transformers and accelerate."
        ) from exc
    model_name = os.getenv("QIMMA_MEDARABIQ_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    device_map = os.getenv("QIMMA_MEDARABIQ_JUDGE_DEVICE_MAP", "auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map=device_map
    )
    model.eval()
    return model, tokenizer


def _context_length(model, tokenizer) -> int:
    candidates = [
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "max_seq_len", None),
        getattr(model.config, "seq_length", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    limits = [
        value
        for value in candidates
        if isinstance(value, int) and 0 < value < 10_000_000
    ]
    if not limits:
        raise ValueError("Could not determine the judge model's context length.")
    return min(limits)


def _judge(question: str, reference: str, prediction: str) -> float:
    model, tokenizer = _load_judge()
    max_input_tokens = _context_length(model, tokenizer) - DEFAULT_MAX_NEW_TOKENS
    if max_input_tokens <= 0:
        raise ValueError("Judge output budget must be smaller than its context length.")
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_judge_prompt(question, reference, prediction),
        },
    ]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0, inputs["input_ids"].shape[1] :]
    return _parse_verdict(tokenizer.decode(generated, skip_special_tokens=True))


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = answer_after_thinking(results[0]).strip()
    if not prediction:
        return {"llm_judge_score": 0.0}
    return {
        "llm_judge_score": _judge(doc["prompt"], doc_to_target(doc), prediction)
    }
