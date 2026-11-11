from lm_eval.tasks.qimma_mizan.utils import (
    ANSWER_LABELS,
    doc_to_target,
    extract_answer,
    process_results,
)


DOC = {"choices": ["one", "two", "three", "four"], "index": [0, 2]}


def test_target_is_canonical_abjad_set():
    assert doc_to_target(DOC) == "أ،ج"
    assert doc_to_target({**DOC, "index": [2, 0, 2]}) == "أ،ج"


def test_extract_answer_accepts_common_separators_and_canonicalizes_order():
    valid = ANSWER_LABELS[:4]
    assert extract_answer("الإجابة <<ج، أ>>", valid) == "أ،ج"
    assert extract_answer("reasoning <<ب>> then <<ج A>>", valid) == "[invalid]"
    assert extract_answer("<<أ،ز>>", valid) == "[invalid]"
    assert extract_answer("أ،ج", valid) == "[invalid]"


def test_extract_answer_only_uses_text_after_closed_thinking():
    valid = ANSWER_LABELS[:4]
    assert extract_answer("<think>ربما <<ب>></think>\n<<أ،ج>>", valid) == "أ،ج"
    assert extract_answer("<think>ربما <<أ>>", valid) == "[invalid]"
    assert extract_answer("<think><<أ>></think>", valid) == "[invalid]"


def test_extract_answer_accepts_untagged_nonthinking_response():
    assert extract_answer("<<أ>>", ANSWER_LABELS[:4]) == "أ"


def test_process_results_scores_sets_not_strings():
    assert process_results(DOC, ["أ،ج"]) == {
        "exact_match": 1.0,
        "set_precision": 1.0,
        "set_recall": 1.0,
        "set_f1": 1.0,
    }
    assert process_results(DOC, ["أ،ب"]) == {
        "exact_match": 0.0,
        "set_precision": 0.5,
        "set_recall": 0.5,
        "set_f1": 0.5,
    }
