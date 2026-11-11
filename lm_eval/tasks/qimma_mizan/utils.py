import re
from collections.abc import Iterable, Sequence
from typing import Any

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter
from lm_eval.tasks._qimma_reasoning import answer_after_thinking


ANSWER_LABELS = tuple("أبجدهوزحطيكلمنسعفصقرشتثخذضظغ")
ANSWER_BLOCK_PATTERN = re.compile(r"<<\s*([^<>]+?)\s*>>")


def _valid_labels(doc: dict[str, Any]) -> tuple[str, ...]:
    if len(doc["choices"]) > len(ANSWER_LABELS):
        raise ValueError("The number of choices exceeds the available Abjad labels")
    return ANSWER_LABELS[: len(doc["choices"])]


def _canonical(labels: Iterable[str], valid_labels: Sequence[str]) -> str:
    selected = set(labels)
    return "،".join(label for label in valid_labels if label in selected)


def extract_answer(response: str, valid_labels: Sequence[str]) -> str:
    """Extract and canonicalize the final structured multi-select answer.

    A valid answer is the final ``<<...>>`` block and may contain labels separated
    by Arabic/ASCII commas or whitespace. Duplicate labels and response order do
    not affect scoring. Any unknown character makes the block invalid, preventing
    free-form reasoning inside brackets from being scored as an answer.
    """

    blocks = ANSWER_BLOCK_PATTERN.findall(answer_after_thinking(response))
    if not blocks:
        return "[invalid]"

    block = blocks[-1].strip()
    allowed_separators = " ،,;؛/|+-\n\t"
    compact = block
    for separator in allowed_separators:
        compact = compact.replace(separator, "")

    valid = set(valid_labels)
    if not compact or any(char not in valid for char in compact):
        return "[invalid]"
    return _canonical(compact, valid_labels)


def doc_to_text(doc: dict[str, Any]) -> str:
    valid_labels = _valid_labels(doc)
    valid_answers = "، ".join(f"<<{label}>>" for label in valid_labels)
    return (
        "لقد تم إعطاؤك سؤالًا حول القانون المغربي. قد يكون للسؤال خيار "
        "صحيح واحد أو عدة خيارات صحيحة. اختر جميع الخيارات الصحيحة فقط، "
        "واكتب حروفها داخل زوج واحد من علامتي الزاوية المزدوجة (<< >>)، "
        "مفصولة بفواصل، في سطر واحد دون أي شرح. "
        f"الحروف المسموحة حصراً هي: {valid_answers}. "
        "مثال لإجابة تحتوي خيارين: <<أ،ج>>.\n\n"
        f"{doc['prompt']}"
    )


def doc_to_target(doc: dict[str, Any]) -> str:
    valid_labels = _valid_labels(doc)
    indices = doc["index"]
    if not indices:
        raise ValueError("MizanQA examples must contain at least one correct choice")
    if any(not isinstance(index, int) or not 0 <= index < len(valid_labels) for index in indices):
        raise ValueError(f"Invalid answer indices: {indices!r}")
    return _canonical((valid_labels[index] for index in indices), valid_labels)


@register_filter("qimma_mizan_answer")
class QimmaMizanAnswerFilter(Filter):
    """Extract the final structured answer as a canonical set of labels."""

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        filtered = []
        for response_set, doc in zip(resps, docs):
            valid_labels = _valid_labels(doc)
            filtered.append(
                [extract_answer(response, valid_labels) for response in response_set]
            )
        return filtered


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """Score exact set equality and per-example set precision/recall/F1."""

    gold = set(doc_to_target(doc).split("،"))
    prediction = results[0] if results else "[invalid]"
    predicted = set() if prediction == "[invalid]" else set(prediction.split("،"))
    true_positives = len(gold & predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_match": float(predicted == gold),
        "set_precision": precision,
        "set_recall": recall,
        "set_f1": f1,
    }
