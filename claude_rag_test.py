"""
VaadAI - Indian Kanoon RAG Pipeline
Flow: User Query -> Claude (generates search query) -> Indian Kanoon API
      -> Claude (synthesises answer) -> User-friendly response

User sees  : plain-language answer + optional follow-up question chips
Logged to  : vaad_ai.log - search plan, citations, doc metadata, timing
"""

import os
import re
import json
import logging
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import quote_plus
from dotenv import load_dotenv
import anthropic

load_dotenv()

# -- Logger -------------------------------------------------------------------
LOG_FILE = "vaad_ai.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("vaadai")

def log_section(title: str):
    log.debug("-" * 60)
    log.debug("  " + title)
    log.debug("-" * 60)


# -- Clients ------------------------------------------------------------------
claude   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
IK_TOKEN = os.getenv("INDIAN_KANOON_API_KEY")
if not IK_TOKEN:
    raise EnvironmentError("INDIAN_KANOON_API_KEY is missing from your .env file")

IK_BASE    = "https://api.indiankanoon.org"
IK_HEADERS = {
    "Authorization": "Token " + IK_TOKEN,
    "Accept": "application/json",
}

DEFAULT_DISCLAIMER = (
    "Warning: This is legal information, not legal advice. Please consult a lawyer for your situation."
)


def split_answer_and_disclaimer(text: str) -> tuple:
    """Split synthesis trailing Warning line from body for separate UI."""
    if "Warning:" in text:
        idx = text.rfind("Warning:")
        body = text[:idx].strip()
        disc = text[idx:].strip()
        if body:
            return body, disc
    return text.strip(), DEFAULT_DISCLAIMER


def build_verified_citation_rows(rag_citations: list, related_cases: list) -> list:
    """
    Rows for API + frontend 'Verified citations' panel. relevance_pct is a simple rank heuristic.
    """
    rows = []
    for i, c in enumerate(rag_citations[:5]):
        pct = max(72, min(96, 94 - i * 4))
        rows.append({
            "title": c.get("title") or "Source",
            "subtitle": (c.get("source") or "").strip(),
            "url": c.get("url") or "",
            "relevance_pct": pct,
            "kind": "context",
        })
    for i, c in enumerate(related_cases[:2]):
        pct = max(65, min(88, 84 - i * 6))
        rows.append({
            "title": c.get("title") or "Case",
            "subtitle": (c.get("source") or "").strip(),
            "url": c.get("url") or "",
            "relevance_pct": pct,
            "kind": "related",
        })
    return rows


# -- Step 0: Query classifier -------------------------------------------------
CLASSIFIER_PROMPT = """You are a query classifier for an Indian legal assistant chatbot.

Classify the user message into one of exactly three categories:

1. LEGAL         - a genuine question about Indian law, rights, court procedures,
                   acts, sections, or legal situations requiring case research.
                   Follow-up questions that dig deeper into a legal topic also count.

2. CONVERSATIONAL - a greeting, thank-you, off-topic remark, or vague statement
                   that does NOT require searching Indian Kanoon.
                   Examples: "thanks", "ok", "who are you?", "reach a lawyer now"

3. OUTOFSCOPE    - a non-Indian or non-legal question entirely outside the domain.
                   Examples: "recipe for dal", "who won IPL?", "write me a poem"

Output ONLY the single word: LEGAL, CONVERSATIONAL, or OUTOFSCOPE.
No explanation, no punctuation."""


def classify_query(user_question: str, history: list = None) -> str:
    """Returns 'LEGAL', 'CONVERSATIONAL', or 'OUTOFSCOPE'."""
    messages = []
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_question})

    resp = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8,
        system=CLASSIFIER_PROMPT,
        messages=messages,
    )
    result = resp.content[0].text.strip().upper()
    log.info("CLASSIFIER | " + repr(result) + " for " + repr(user_question))
    return result if result in ("LEGAL", "CONVERSATIONAL", "OUTOFSCOPE") else "LEGAL"


CONVERSATIONAL_SYSTEM = """You are VaadAI, a friendly Indian legal information assistant.
The user has sent a non-legal or conversational message.
Respond warmly and briefly in 2-3 sentences max.
If they want to reach a lawyer, suggest searching for a local advocate
or contacting their state bar association - but do NOT mention any cases.
Never pretend to be a general-purpose assistant.
End with: "Feel free to ask me any Indian legal question!" """


def handle_conversational(user_question: str, history: list = None) -> str:
    """Handle greetings, thanks, vague requests without hitting IK at all."""
    messages = []
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_question})

    resp = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        system=CONVERSATIONAL_SYSTEM,
        messages=messages,
    )
    return resp.content[0].text.strip()


# -- Step 1: Search planner ---------------------------------------------------
SEARCH_PLANNER_PROMPT = """You are a legal search specialist for Indian law.
Given a user's legal question, output a JSON object (and NOTHING else) with:

{
  "formInput": "<concise Indian Kanoon search query>",
  "doctypes":  "<one of: supremecourt | highcourts | judgments | laws | tribunals | (blank for all)>",
  "reasoning": "<one sentence on why you chose this query>"
}

Rules:
- Keep formInput short (3-8 words), keyword-rich, suited for Indian Kanoon full-text search.
- Use ANDD / ORR / NOTT operators where useful (they are case-sensitive).
- Pick doctypes that best fits the question; leave blank if unsure.
- Output only valid JSON - no markdown fences, no preamble."""


def plan_search(user_question: str) -> dict:
    """Ask Claude to craft the best Indian Kanoon search query."""
    resp = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=SEARCH_PLANNER_PROMPT,
        messages=[{"role": "user", "content": user_question}],
    )
    raw  = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    plan = json.loads(raw)
    log.info("SEARCH PLAN | formInput=" + repr(plan["formInput"]) + "  doctypes=" + repr(plan.get("doctypes")))
    log.debug("SEARCH PLAN REASONING | " + str(plan.get("reasoning")))
    return plan


# -- Step 2: IK search --------------------------------------------------------
def search_indian_kanoon(form_input: str, doctypes: str = "", pagenum: int = 0, max_results: int = 10) -> list:
    """Search IK sorted by most recent. Returns docs list for RAG context."""
    query = form_input + " sortby:mostrecent"
    if doctypes:
        query += " doctypes:" + doctypes

    t0   = time.time()
    resp = requests.post(
        IK_BASE + "/search/",
        headers=IK_HEADERS,
        data={"formInput": query, "pagenum": pagenum},
        timeout=20,
    )
    resp.raise_for_status()
    data    = resp.json()
    elapsed = time.time() - t0

    docs = data.get("docs", [])[:max_results]
    log.info("IK SEARCH | found=" + str(data.get("found", "?")) + "  returned=" + str(len(docs)) + "  time=" + str(round(elapsed, 2)) + "s")
    return docs


# -- Step 3: Fetch doc text ---------------------------------------------------
MAX_CHARS_PER_DOC = 3000


def fetch_doc_text(tid: int, form_input: str) -> str:
    """Fetch highlighted fragment; fall back to full doc."""
    try:
        resp = requests.post(
            IK_BASE + "/docfragment/" + str(tid) + "/",
            headers=IK_HEADERS,
            data={"formInput": form_input},
            timeout=20,
        )
        resp.raise_for_status()
        headline = resp.json().get("headline", "")
        if headline:
            return headline[:MAX_CHARS_PER_DOC]
    except Exception as e:
        log.warning("FRAGMENT failed | tid=" + str(tid) + "  err=" + str(e))

    try:
        resp = requests.post(IK_BASE + "/doc/" + str(tid) + "/", headers=IK_HEADERS, timeout=20)
        resp.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", resp.json().get("doc", ""))
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_CHARS_PER_DOC]
    except Exception as e:
        log.warning("FULL DOC failed | tid=" + str(tid) + "  err=" + str(e))
        return ""


def _fetch_doc_for_context(doc: dict, form_input: str) -> tuple:
    """Single-doc fetch for parallel build_context."""
    tid   = doc.get("tid")
    title = doc.get("title", "Unknown")
    src   = doc.get("docsource", "")
    url   = "https://indiankanoon.org/doc/" + str(tid) + "/"
    text  = fetch_doc_text(tid, form_input)
    return tid, title, src, url, text


def build_context(docs: list, form_input: str) -> tuple:
    """Fetch text for each doc in parallel; return context string + citations metadata."""
    if not docs:
        return "", []

    max_workers = min(10, len(docs))
    context_parts = []
    citations     = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_fetch_doc_for_context, d, form_input) for d in docs]
        for fut in futures:
            tid, title, src, url, text = fut.result()
            if not text:
                continue
            context_parts.append("### " + title + " [" + src + "]\n" + text)
            citations.append({"title": title, "source": src, "url": url, "tid": tid})
            log.info("CITATION | " + repr(title) + "  " + url)

    return "\n\n".join(context_parts), citations


# -- Step 4: Answer synthesis -------------------------------------------------
ANSWER_SYNTHESISER_PROMPT = """You are VaadAI, a legal information assistant for India helping ordinary people understand their rights.

You will receive a user's legal question and verified excerpts from Indian court judgments and statutes.

LENGTH RULES (strict):
- Your entire answer must be 6-10 lines maximum. No exceptions.
- One direct opening sentence answering the question.
- Then 3-5 bullet points covering the key points. Each bullet: one line only.
- Then the disclaimer line. That is it.

CONTENT RULES:
- Plain, simple language - as if texting a friend. Zero jargon.
- If the answer comes directly from a specific law, mention only its short name and section number.
- If the law has been renamed (e.g. CrPC -> BNSS), add it in brackets: "Section 482 CrPC (now Section 528 BNSS)".
- NEVER include case names, citation numbers, document IDs, or URLs.
- If context is insufficient, say so in one line and name the type of lawyer to consult.

End with exactly this line:
Warning: This is legal information, not legal advice. Please consult a lawyer for your situation."""


def synthesise_answer(user_question: str, context: str, history: list = None) -> str:
    """Ask Claude for a plain-language answer (no JSON)."""
    t0       = time.time()
    messages = []
    for turn in (history or []):
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": "USER QUESTION:\n" + user_question + "\n\nINDIAN KANOON CONTEXT:\n" + context,
    })

    resp = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=ANSWER_SYNTHESISER_PROMPT,
        messages=messages,
    )
    elapsed = time.time() - t0
    answer  = resp.content[0].text.strip()
    log.info("SYNTHESIS | tokens_in=" + str(resp.usage.input_tokens) + "  tokens_out=" + str(resp.usage.output_tokens) + "  time=" + str(round(elapsed, 2)) + "s")
    return answer


# -- Step 4a: Follow-up recommendations (separate call — reliable JSON, faster than bloating synthesis) --
FOLLOW_UP_SYSTEM = """You generate follow-up questions for VaadAI, an Indian legal information chatbot.

You receive: the user's question, the assistant's answer (plain text), and optional excerpts from Indian Kanoon.

Output exactly 3 follow-up questions the USER might tap next.

Rules:
- Each question must be about Indian law / procedure only, and must be something this assistant can answer
  using statutes and reported cases (not fact-specific legal advice for a real person's undisclosed case).
- Keep each question under 12 words. Natural, conversational Hindi-English mix is OK if the user did that.
- Tie questions to the topic just explained; prefer narrower, quicker-to-answer questions over huge new topics.
- One clarification-style, one deeper rule/procedure, one practical "what happens if / what are penalties" style — when it fits.
- Do not repeat the original question verbatim.

Output ONLY a JSON array of 3 strings, e.g. ["...","...","..."]
No markdown fences, no other keys, no numbering outside the JSON."""


def generate_follow_up_questions(
    user_question: str,
    answer_text: str,
    context: str = "",
    history: list = None,
) -> list:
    """
    Second Claude call: short, focused follow-ups grounded in Q + A + optional RAG context.
    """
    ctx = (context or "")[:4500]
    block = (
        "ORIGINAL USER QUESTION:\n" + user_question.strip()
        + "\n\nASSISTANT ANSWER:\n" + answer_text.strip()
        + ("\n\nINDIAN KANOON EXCERPTS (for grounding):\n" + ctx if ctx.strip() else "")
    )
    messages = []
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": block})

    t0 = time.time()
    resp = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=220,
        system=FOLLOW_UP_SYSTEM,
        messages=messages,
    )
    elapsed = time.time() - t0
    raw = resp.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    log.info(
        "FOLLOW-UPS | tokens_out=" + str(resp.usage.output_tokens) + "  time=" + str(round(elapsed, 2)) + "s"
    )

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("not a list")
        out = [s.strip() for s in parsed if isinstance(s, str) and len(s.strip()) > 3][:3]
    except Exception as e:
        log.warning("FOLLOW-UPS JSON parse failed: " + str(e) + " | raw=" + repr(raw[:200]))
        return []

    log.info("SUGGESTIONS | " + str(out))
    return out


# -- Step 4b: Fetch top 2 relevant cases via IK API ---------------------------
def fetch_top_cases(user_question: str) -> tuple:
    """
    Extract keywords, search IK API (authenticated - never blocked),
    return top 2 distinct cases and a public browse URL.
    """
    # Claude extracts tight keywords
    kw_resp = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=32,
        system=(
            "Extract a 3-6 word keyword phrase describing the SPECIFIC legal issue "
            "(e.g. 'FIR quash Section 482 CrPC', 'eviction non-payment rent', "
            "'maintenance Section 125 CrPC'). Output ONLY the phrase, nothing else."
        ),
        messages=[{"role": "user", "content": user_question}],
    )
    keywords   = kw_resp.content[0].text.strip()
    search_url = "https://indiankanoon.org/search/?formInput=" + quote_plus(keywords + " sortby:mostrecent")
    log.info("CASE KEYWORDS | " + repr(keywords))

    query = keywords + " sortby:mostrecent doctypes:judgments"
    try:
        resp = requests.post(
            IK_BASE + "/search/",
            headers=IK_HEADERS,
            data={"formInput": query, "pagenum": 0},
            timeout=20,
        )
        resp.raise_for_status()
        docs = resp.json().get("docs", [])
    except Exception as e:
        log.warning("CASE SEARCH failed: " + str(e))
        return [], search_url

    def title_similarity(a: str, b: str) -> float:
        a, b = a.lower().strip(), b.lower().strip()
        if a == b:
            return 1.0
        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s) - 1))
        bg_a, bg_b = bigrams(a), bigrams(b)
        if not bg_a or not bg_b:
            return 0.0
        return 2 * len(bg_a & bg_b) / (len(bg_a) + len(bg_b))

    cases = []
    for d in docs:
        tid   = d.get("tid")
        title = d.get("title", "Unknown")
        src   = d.get("docsource", "")
        url   = "https://indiankanoon.org/doc/" + str(tid) + "/"

        if any(title_similarity(title, c["title"]) >= 0.97 for c in cases):
            log.debug("SKIPPED duplicate | " + repr(title))
            continue

        cases.append({"title": title, "source": src, "url": url})
        if len(cases) == 2:
            break

    log.info("CASES SELECTED | " + str([c["title"] for c in cases]))
    return cases, search_url


# -- Orchestrator -------------------------------------------------------------
def rag_query(user_question: str, history: list = None) -> dict:
    """
    Full pipeline. Returns dict with answer body, disclaimer, citations[], suggestions, more_cases_url.
    history: list of {role, content} dicts from the frontend session.
    """
    run_id = datetime.now().strftime("%H%M%S")
    log_section("RUN " + run_id + " | " + user_question)

    empty_extras = {"citations": [], "more_cases_url": None, "disclaimer": None}

    # 0. Classify - skip IK entirely for non-legal messages
    query_type = classify_query(user_question, history)

    if query_type == "OUTOFSCOPE":
        log.info("OUTOFSCOPE - skipping IK search")
        return {
            "answer": ("I am VaadAI, an Indian legal information assistant. "
                       "I can only help with questions about Indian law, court procedures, and legal rights.\n\n"
                       "Feel free to ask me any Indian legal question!"),
            "suggestions": [],
            **empty_extras,
        }

    if query_type == "CONVERSATIONAL":
        log.info("CONVERSATIONAL - skipping IK search")
        return {
            "answer": handle_conversational(user_question, history),
            "suggestions": [],
            **empty_extras,
        }

    # 1. Plan search
    search_plan = plan_search(user_question)
    form_input  = search_plan["formInput"]
    doctypes    = search_plan.get("doctypes", "")

    # 2. Search IK for RAG context
    docs = search_indian_kanoon(form_input, doctypes)
    if not docs:
        log.warning("No documents returned from IK search.")
        return {
            "answer": (
                "I was not able to find relevant legal information for your question right now. "
                "Please try rephrasing, or consult a local lawyer directly."
            ),
            "suggestions": [],
            "citations": [],
            "more_cases_url": None,
            "disclaimer": DEFAULT_DISCLAIMER,
        }

    # 3. Build context (parallel doc fetch)
    context, rag_citations = build_context(docs, form_input)
    if not context.strip():
        context = "No usable context could be retrieved."
    log.info("CONTEXT BUILT | docs_used=" + str(len(rag_citations)) + "  total_chars=" + str(len(context)))

    # 4. Synthesise answer (plain text)
    raw_answer = synthesise_answer(user_question, context, history=history)
    answer_body, disclaimer = split_answer_and_disclaimer(raw_answer)

    # 5. Follow-up questions + related cases in parallel (same wall-clock as one of them)
    suggestions = []
    relevant_cases = []
    ik_search_url   = None
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_sug = ex.submit(
            generate_follow_up_questions, user_question, raw_answer, context, history
        )
        fut_cases = ex.submit(fetch_top_cases, user_question)
        try:
            suggestions = fut_sug.result()
        except Exception as e:
            log.warning("follow_up_questions failed: " + str(e))
            suggestions = []
        try:
            relevant_cases, ik_search_url = fut_cases.result()
        except Exception as e:
            log.warning("fetch_top_cases failed: " + str(e))
            relevant_cases, ik_search_url = [], None

    citation_rows = build_verified_citation_rows(rag_citations, relevant_cases)

    log.info("RUN " + run_id + " COMPLETE")
    return {
        "answer": answer_body,
        "disclaimer": disclaimer,
        "citations": citation_rows,
        "more_cases_url": ik_search_url,
        "suggestions": suggestions,
    }


# -- Test ---------------------------------------------------------------------
if __name__ == "__main__":
    queries = [
        "Can FIR be quashed under Section 482 CrPC? What are the grounds courts have accepted?",
        "Landlord is trying to evict my client without 30-day notice. Which Rent Control Act sections apply in Maharashtra?",
        "How is maintenance calculated under Section 125 CrPC? What factors does the court consider?",
    ]
    print("VaadAI | Technical logs -> vaad_ai.log\n" + "=" * 60)
    for i, q in enumerate(queries, 1):
        print("\nQuestion " + str(i) + ": " + q + "\n")
        result = rag_query(q)
        print(result["answer"])
        if result.get("disclaimer"):
            print("\n[" + result["disclaimer"] + "]")
        if result.get("citations"):
            print("\nCitations:", len(result["citations"]))
        if result["suggestions"]:
            print("\nSuggested follow-ups:")
            for s in result["suggestions"]:
                print("  - " + s)
        print("\n" + "-" * 60)
    print("\nDone. Check vaad_ai.log for full technical details.")
