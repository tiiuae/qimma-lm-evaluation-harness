"""Local LLM-as-judge scoring for QIMMA AraLegalEval."""

import re
from typing import Any

from lm_eval.tasks._qimma_judge import load_judge as _load_judge
from lm_eval.tasks._qimma_reasoning import answer_after_thinking
from lm_eval.tasks.qimma_arablegaleval.utils import doc_to_target


DEFAULT_MAX_NEW_TOKENS = 512
_VERDICT_RE = re.compile(r"\[\[\s*([01])\s*\]\]")

_SYSTEM_PROMPT = """أنت محكّم خبير في الأسئلة القانونية العربية. مهمتك تحديد ما إذا كانت إجابة النموذج مطابقة في المعنى للإجابة المرجعية.
الإجابة المرجعية هي المعيار الوحيد للصواب. لا تستخدم معرفتك القانونية لقبول إجابة تختلف عن المرجع، حتى لو بدت لك معقولة.

قواعد التقييم:

1. المطابقة في المعنى لا في اللفظ: اقبل المرادفات والصياغات القانونية المكافئة والاختصارات والترجمة بين العربية والإنجليزية للمفهوم القانوني نفسه.
2. يجب أن يتطابق الحكم أو المفهوم أو القاعدة أو الجهة أو الإجراء القانوني المحدد مع المرجع. إذا ذكرت الإجابة مفهوماً أو حكماً أو جهة أو إجراءً مختلفاً، فهي خاطئة حتى لو كان قريباً منه أو من المجال القانوني نفسه.
3. إذا كان السؤال يتطلب عدة عناصر أو شروط أو آثار قانونية، فيجب أن تكون جميع العناصر المطلوبة صحيحة ومكتملة. إذا كان عنصر جوهري واحد خاطئاً أو ناقصاً فالدرجة 0. لا توجد درجات جزئية.
4. الإجابة الأعم من المطلوب أو المبهمة تُعتبر خاطئة إذا كان المرجع يحدد حكماً أو مفهوماً أو جهة أو إجراءً بعينه.
5. ذكر عدة احتمالات متعارضة أو التردد بينها يُعتبر خاطئاً، ما لم تكن الإجابة المرجعية نفسها تتطلب تعدد الاحتمالات.
6. لا تعاقب على اختلاف الأسلوب أو على تفاصيل إضافية صحيحة. أما الإضافات الخاطئة أو المتناقضة مع المرجع فتجعل الدرجة 0.
7. عند الشك، امنح 0.

صيغة الرد:
اكتب الحكم النهائي فقط دون مقارنة أو تعليل أو شرح أو أي نص إضافي.
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
