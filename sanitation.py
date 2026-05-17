"""
Relevance sanity-check + general-knowledge fallback for the RAG pipeline.

Why this exists:
- The pipeline can produce an answer that is technically synthesised but
  doesn't actually address the question — usually when Indian Kanoon
  returned no useful context, or the retrieved docs were tangential.
- For genuinely legal questions we'd rather give the user something useful
  from Claude's own knowledge than an empty "couldn't find" message. The
  upstream classifier already shunts truly off-topic messages elsewhere, so
  by the time we get here a fallback answer is appropriate.

Token strategy (cheap by design):
- Heuristic phrase scan first (free). Two or more bail-phrases => IRRELEVANT,
  no LLM call.
- If the heuristic is ambiguous, ask Haiku for a one-word verdict
  (~1 output token, ~150 input tokens after truncation).
- Fallback synthesis runs Sonnet only when the answer was judged irrelevant.
"""

import os
import time
import logging
import anthropic

log = logging.getLogger("vaadai")

_claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

RELEVANCE_MODEL = "claude-haiku-4-5-20251001"
FALLBACK_MODEL  = "claude-sonnet-4-20250514"

# Telltale phrases the synthesiser uses when it knows it's bailing. Hits in
# the answer text are a strong "irrelevant" signal without spending tokens.
_BAIL_PHRASES = (
    "indian kanoon doesn't have",
    "indian kanoon does not have",
    "i cannot find",
    "i could not find",
    "i was not able to find",
    "i don't have",
    "i do not have",
    "unable to find",
    "no relevant",
    "context is insufficient",
    "no usable context",
    "insufficient context",
    "couldn't find",
)

_RELEVANCE_SYSTEM = (
    "You judge whether an assistant's answer actually addresses the user's "
    "legal question. Output EXACTLY one word: RELEVANT or IRRELEVANT.\n"
    "RELEVANT = answer contains usable legal information responsive to the "
    "question.\n"
    "IRRELEVANT = answer says it can't find / doesn't know, gives generic "
    "deflection, or discusses a different topic."
)


def is_answer_relevant(question: str, answer: str) -> bool:
    """Cheap relevance check. True if `answer` actually addresses `question`."""
    if not answer or len(answer.strip()) < 20:
        return False

    lower = answer.lower()
    bail_hits = sum(1 for p in _BAIL_PHRASES if p in lower)
    if bail_hits >= 2:
        log.info("RELEVANCE | heuristic IRRELEVANT (bail_hits=%d)", bail_hits)
        return False

    # Single weak signal — bring in Haiku for a one-token verdict. Inputs are
    # aggressively truncated so the call stays small.
    q_short = (question or "")[:300]
    a_short = (answer or "")[:600]

    try:
        t0 = time.time()
        resp = _claude.messages.create(
            model=RELEVANCE_MODEL,
            max_tokens=4,
            system=_RELEVANCE_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"QUESTION:\n{q_short}\n\nANSWER:\n{a_short}",
            }],
        )
        elapsed = time.time() - t0
        verdict = (resp.content[0].text or "").strip().upper()
        log.info(
            "RELEVANCE | Claude verdict=%r in=%d out=%d time=%.2fs",
            verdict,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            elapsed,
        )
        return verdict.startswith("RELEVANT")
    except Exception as e:
        # Fail-open: a flaky relevance check shouldn't punish the user by
        # forcing the fallback path. Trust the original answer instead.
        log.warning("RELEVANCE check failed, defaulting to relevant: %s", e)
        return True


_FALLBACK_SYSTEM = """You are VaadAI, a legal information assistant for India helping ordinary people understand their rights.

Indian Kanoon returned no useful context for this question, so answer from your own general knowledge of Indian law.

LENGTH RULES (strict):
- Entire answer must be 6-10 lines maximum.
- One direct opening sentence answering the question.
- 3-5 bullet points covering the key points. One line each.
- Then the disclaimer line.

CONTENT RULES:
- Plain, simple language - as if texting a friend. Zero jargon.
- If a specific law applies, mention only its short name and section number.
- If renamed (e.g. CrPC -> BNSS), add it in brackets: "Section 482 CrPC (now Section 528 BNSS)".
- NEVER invent case names, citation numbers, or specific judgments. If you're unsure, say so.
- NEVER include URLs or document IDs.
- If the question is not about Indian law, say so in one line and name the type of professional to consult.

End with exactly this line:
Warning: This is general legal information, not legal advice. Please consult a lawyer for your situation."""


def fallback_answer_from_claude_knowledge(user_question: str, history: list = None) -> str:
    """
    Synthesise a plain-language answer from Claude's general knowledge, used
    when the RAG pipeline couldn't produce a useful response.
    """
    messages = []
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_question})

    t0 = time.time()
    resp = _claude.messages.create(
        model=FALLBACK_MODEL,
        max_tokens=400,
        system=_FALLBACK_SYSTEM,
        messages=messages,
    )
    elapsed = time.time() - t0
    answer = (resp.content[0].text or "").strip()
    log.info(
        "FALLBACK | tokens_in=%d tokens_out=%d time=%.2fs",
        resp.usage.input_tokens,
        resp.usage.output_tokens,
        elapsed,
    )
    return answer
