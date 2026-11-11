from typing import Any


def doc_to_text(doc: dict[str, Any]) -> str:
    return doc["prompt"]


def doc_to_target(doc: dict[str, Any]) -> str:
    choices = doc["choices"]
    if isinstance(choices, str):
        return choices
    return choices[doc.get("index", 0)]
