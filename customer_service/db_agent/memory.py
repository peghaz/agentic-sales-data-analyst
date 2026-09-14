"""Bounded, session-scoped context for analytical follow-up questions."""

from __future__ import annotations

import json

from .types import QueryTrace

MAX_HISTORY_TURNS = 3
MAX_QUESTION_CHARS = 600
MAX_ANSWER_CHARS = 1600
MAX_EVIDENCE_CHARS = 6000


def history_messages(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return completed exchanges in chronological model-message order."""

    messages: list[dict[str, str]] = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        messages.extend(
            [
                {"role": "user", "content": turn["question"]},
                {"role": "assistant", "content": turn["answer"]},
            ]
        )
    return messages


def evidence_message(history: list[dict[str, str]]) -> str | None:
    """Expose the latest bounded query evidence for references such as 'those'."""

    if not history:
        return None
    evidence = history[-1].get("evidence", "")
    if not evidence:
        return None
    return (
        "Evidence from the most recent completed analysis follows. It is context, "
        "not a substitute for querying when the user asks for new data. "
        f"If a reference cannot be resolved from this and the recent exchanges, ask "
        f"the user to clarify.\n{evidence}"
    )


def remember_turn(
    history: list[dict[str, str]],
    question: str,
    answer: str,
    traces: list[QueryTrace],
) -> list[dict[str, str]]:
    """Keep only completed turns and a small sample of validated evidence."""

    successful = [trace for trace in traces if not trace.error]
    evidence = [
        {
            "purpose": trace.purpose,
            "sql": trace.sql[:2000],
            "row_count": trace.row_count,
            "truncated": trace.truncated,
            "rows": trace.rows[:10],
        }
        for trace in successful
    ]
    encoded = json.dumps(evidence, ensure_ascii=False, default=str)
    if len(encoded) > MAX_EVIDENCE_CHARS:
        encoded = encoded[:MAX_EVIDENCE_CHARS] + "... [evidence shortened]"
    prior_evidence = history[-1].get("evidence", "") if history else ""
    turn = {
        "question": question[:MAX_QUESTION_CHARS],
        "answer": answer[:MAX_ANSWER_CHARS],
        "evidence": encoded if successful else prior_evidence,
    }
    return [*history[-(MAX_HISTORY_TURNS - 1) :], turn]
