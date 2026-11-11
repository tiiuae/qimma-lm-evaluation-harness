import re
from collections.abc import Iterable, Sequence
from typing import Any

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


ANSWER_LABELS = tuple("أبجد")
ANSWER_PATTERN = re.compile(r"<<\s*([أبجد])\s*>>")
OPTION_COLUMNS = tuple(f"option_{index}" for index in range(1, 5))


def doc_to_text(doc: dict[str, Any]) -> str:
    """Build the MCQ prompt from Alyah's query and four option columns."""
    options = [doc[column] for column in OPTION_COLUMNS]
    valid_answers = "، ".join(f"<<{label}>>" for label in ANSWER_LABELS)
    formatted_options = "\n".join(
        f"{label}) {option}" for label, option in zip(ANSWER_LABELS, options)
    )
    return (
        "اختر الإجابة الصحيحة، واكتب حرف الخيار الصحيح فقط بين علامتي زاوية "
        "مزدوجتين (<< >>) في سطر واحد، دون إضافة أي شرح أو نص آخر. "
        f"الخيارات المسموح بها حصراً هي: {valid_answers}.\n\n"
        f"{doc['query']}\n{formatted_options}"
    )


def doc_to_target(doc: dict[str, Any]) -> str:
    correct_answer = int(doc["correct_answer"])
    if not 1 <= correct_answer <= len(ANSWER_LABELS):
        raise ValueError(
            f"correct_answer must be between 1 and {len(ANSWER_LABELS)}, "
            f"got {correct_answer}"
        )
    return ANSWER_LABELS[correct_answer - 1]


@register_filter("qimma_alyah_answer")
class QimmaAlyahAnswerFilter(Filter):
    """Extract the last structured answer, allowing reasoning to precede it."""

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        filtered = []
        valid_labels = set(ANSWER_LABELS)
        for response_set, _doc in zip(resps, docs):
            extracted = []
            for response in response_set:
                matches = ANSWER_PATTERN.findall(
                    response if isinstance(response, str) else ""
                )
                valid_matches = [match for match in matches if match in valid_labels]
                extracted.append(valid_matches[-1] if valid_matches else "[invalid]")
            filtered.append(extracted)
        return filtered
