"""Closed-model LLM judge for QIMMA AraLegalEval, hosted by OpenRouter."""

import os
import re
from functools import lru_cache
from threading import BoundedSemaphore
from typing import Any

from lm_eval.tasks._qimma_reasoning import answer_after_thinking
from lm_eval.tasks.qimma_arablegaleval.utils import doc_to_target


JUDGE_MODEL_NAME = "google/gemini-3.7-flash"
DEFAULT_MAX_CONCURRENT_REQUESTS = 16
JUDGE_MAX_TOKENS = 2048
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_VERDICT_RE = re.compile(r"\[\[\s*([01])\s*\]\]")

_SYSTEM_PROMPT = """أنت محكّم خبير في القانون العربي. قيّم إجابة النموذج اعتماداً فقط على السؤال والإجابة المرجعية.

قواعد التقييم:
- امنح 1 إذا كانت الإجابة صحيحة في المعنى وتغطي الجواب القانوني المطلوب، حتى مع اختلاف الصياغة.
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
    if not matches:
        return 0.5
    if len(set(matches)) != 1:
        return 0.0
    return float(matches[-1])


@lru_cache(maxsize=1)
def _request_semaphore() -> BoundedSemaphore:
    return BoundedSemaphore(DEFAULT_MAX_CONCURRENT_REQUESTS)


@lru_cache(maxsize=1)
def _openrouter_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ModuleNotFoundError(
            "qimma_arablegaleval_judge_closed requires the openai package."
        ) from exc
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set for the closed judge")
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        max_retries=5,
        timeout=120.0,
    )


def _judge(question: str, reference: str, prediction: str) -> float:
    model_name = os.getenv("QIMMA_LEGAL_CLOSED_JUDGE_MODEL", JUDGE_MODEL_NAME)
    with _request_semaphore():
        response = _openrouter_client().chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=JUDGE_MAX_TOKENS,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_judge_prompt(question, reference, prediction),
                },
            ],
        )
    return _parse_verdict(response.choices[0].message.content or "")


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = answer_after_thinking(results[0]).strip()
    if not prediction:
        return {"llm_judge_score": 0.0}
    return {
        "llm_judge_score": _judge(doc["prompt"], doc_to_target(doc), prediction)
    }
