"""Shared response-boundary handling for QIMMA task scorers."""

import re


_THINKING_OPEN = re.compile(
    r"(?:"
    r"<(?:think|thinking|reasoning|analysis)\b[^>]*>"
    r"|<ifm\|think\s*>"
    r"|<\|channel\|>\s*(?:analysis|reasoning)\s*<\|message\|>"
    r"|<\|(?:analysis|reasoning)\|>"
    r")",
    flags=re.IGNORECASE,
)
_THINKING_CLOSE = re.compile(
    r"(?:"
    r"</(?:think|thinking|reasoning|analysis)\s*>"
    r"|</ifm\|think\s*>"
    r")",
    flags=re.IGNORECASE,
)
_FINAL_BOUNDARY = re.compile(
    r"(?:"
    r"<\|channel\|>\s*final\s*<\|message\|>"
    r"|<\|final\|>"
    r")",
    flags=re.IGNORECASE,
)


def answer_after_thinking(response: object) -> str:
    """Return only answer text following the final reasoning boundary."""
    if not isinstance(response, str):
        return ""

    openings = list(_THINKING_OPEN.finditer(response))
    boundaries = list(_THINKING_CLOSE.finditer(response)) + list(
        _FINAL_BOUNDARY.finditer(response)
    )
    if not openings and not boundaries:
        return response
    if not boundaries:
        return ""

    final_boundary = max(boundaries, key=lambda match: match.start())
    if openings and openings[-1].start() > final_boundary.start():
        return ""
    return response[final_boundary.end() :]
