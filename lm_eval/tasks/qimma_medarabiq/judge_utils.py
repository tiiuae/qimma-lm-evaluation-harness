"""LLM-as-judge scoring for the generative QIMMA MedArabiQ subset."""

import re
from typing import Any

from lm_eval.tasks._qimma_judge import load_judge as _load_judge
from lm_eval.tasks._qimma_reasoning import answer_after_thinking
from lm_eval.tasks.qimma_medarabiq.qa_utils import doc_to_target


DEFAULT_MAX_NEW_TOKENS = 512
_VERDICT_RE = re.compile(r"\[\[\s*([01])\s*\]\]")

_SYSTEM_PROMPT = """أنت محكّم خبير في الأسئلة الطبية العربية. مهمتك تحديد ما إذا كانت إجابة النموذج مطابقة في المعنى للإجابة المرجعية.
الإجابة المرجعية هي المعيار الوحيد للصواب. لا تستخدم معرفتك الطبية لقبول إجابة تختلف عن المرجع، حتى لو بدت لك معقولة.

قواعد التقييم:

1. المطابقة في المعنى لا في اللفظ: اقبل المرادفات والاختصارات والترجمة بين العربية والإنجليزية والأسماء العلمية لنفس الشيء (مثال: VZV = فيروس الحماق النطاقي).
2. يجب أن يتطابق الكيان المحدد: إذا ذكرت الإجابة مرضاً أو ممرضاً أو دواءً أو فئة دوائية أو بنية تشريحية أو مادة مختلفة عن المرجع، فهي خاطئة حتى لو كانت من نفس العائلة أو قريبة منها.
   أمثلة: فيروس الهربس البسيط ≠ فيروس الحماق النطاقي، البكتيريا ≠ فيروس الورم الحليمي البشري، البنسلين ≠ السيفالوسبورينات، الانصباب ≠ الإفرازات.
3. الأسئلة متعددة الفراغات: قيّم كل فراغ على حدة. يجب أن تكون جميع الفراغات صحيحة. إذا كان فراغ واحد خاطئاً أو ناقصاً فالدرجة 0. لا توجد درجات جزئية.
4. الإجابة الأعم من المطلوب (مثل "بكتيريا" أو "عدوى" بدلاً من ممرض محدد) تُعتبر خاطئة.
5. ذكر عدة احتمالات أو التردد بينها يُعتبر خاطئاً.
6. لا تعاقب على اختلاف الأسلوب أو على تفاصيل إضافية صحيحة. أما الإضافات الخاطئة أو المتناقضة مع المرجع فتجعل الدرجة 0.
7. عند الشك، امنح 0.

صيغة الرد:
اكتب سطراً واحداً قصيراً يقارن كل فراغ بالمرجع، ثم الحكم النهائي في سطر مستقل:
المقارنة: <فراغ 1: مطابق/غير مطابق، فراغ 2: ...>
الحكم: [[1]] أو [[0]]"""


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
