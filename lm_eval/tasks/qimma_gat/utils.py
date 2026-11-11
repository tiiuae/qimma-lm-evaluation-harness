import re
from collections.abc import Iterable, Sequence
from typing import Any

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter
from lm_eval.tasks._qimma_reasoning import answer_after_thinking


ANSWER_LABELS = tuple("أبجدهوزحطيكلمنسعفصقرشتثخذضظغ")
ANSWER_PATTERN = re.compile(r"<<\s*([أبجدهوزحطيكلمنسعفصقرشتثخذضظغ])\s*>>")


def doc_to_text(doc: dict[str, Any]) -> str:
    if len(doc["choices"]) > len(ANSWER_LABELS):
        raise ValueError("The number of choices exceeds the available Abjad labels")
    valid_labels = ANSWER_LABELS[: len(doc["choices"])]
    valid_answers = "، ".join(f"<<{label}>>" for label in valid_labels)
    return (
        "اختر الإجابة الصحيحة، واكتب حرف الخيار الصحيح فقط بين علامتي زاوية "
        "مزدوجتين (<< >>) في سطر واحد، دون إضافة أي شرح أو نص آخر. "
        f"الخيارات المسموح بها حصراً هي: {valid_answers}.\n\n"
        f"{doc['prompt']}"
    )


def doc_to_target(doc: dict[str, Any]) -> str:
    return ANSWER_LABELS[doc["index"]]


@register_filter("qimma_gat_answer")
class QimmaGATAnswerFilter(Filter):
    """Extract the last structured answer, allowing reasoning to precede it."""

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        filtered = []
        for response_set, doc in zip(resps, docs):
            valid_labels = set(ANSWER_LABELS[: len(doc["choices"])])
            extracted = []
            for response in response_set:
                matches = ANSWER_PATTERN.findall(answer_after_thinking(response))
                valid_matches = [match for match in matches if match in valid_labels]
                extracted.append(valid_matches[-1] if valid_matches else "[invalid]")
            filtered.append(extracted)
        return filtered
