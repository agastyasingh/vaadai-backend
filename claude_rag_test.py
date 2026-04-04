"""
VaadAI – Indian Kanoon RAG Pipeline
Flow: User Query → Claude (generates search query) → Indian Kanoon API
      → Claude (synthesises answer) → User-friendly response

User sees  : plain-language answer only
Logged to  : vaad_ai.log — search plan, citations, doc metadata, timing
"""

import os
import json
import logging
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ── Logger (all technical details go here, not to stdout) ─────────────────────
LOG_FILE = "vaad_ai.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("vaadai")

def log_section(title: str):
    log.debug("─" * 60)
    log.debug(f"  {title}")
    log.debug("─" * 60)


# ── Clients ────────────────────────────────────────────────────────────────────
claude   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
IK_TOKEN = os.getenv("INDIAN_KANOON_API_KEY")
if not IK_TOKEN:
    raise EnvironmentError("INDIAN_KANOON_API_KEY is missing from your .env file")

IK_BASE    = "https://api.indiankanoon.org"
IK_HEADERS = {
    "Authorization": f"Token {IK_TOKEN}",
    "Accept": "application/json",
}


# ── Step 1: Claude → optimised IK search query ────────────────────────────────
SEARCH_PLANNER_PROMPT = """
You are a legal search specialist for Indian law.
Given a user's legal question, output a JSON object (and NOTHING else) with:

{
  "formInput": "<concise Indian Kanoon search query>",
  "doctypes":  "<one of: supremecourt | highcourts | judgments | laws | tribunals | (blank for all)>",
  "reasoning": "<one sentence on why you chose this query>"
}

Rules:
- Keep formInput short (3-8 words), keyword-rich, and suited for Indian Kanoon full-text search.
- Use ANDD / ORR / NOTT operators where useful (they are case-sensitive).
- Pick doctypes that best fits the question; leave blank if unsure.
- Output only valid JSON — no markdown fences, no preamble.
"""

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
    log.info(f"SEARCH PLAN | formInput={plan['formInput']!r}  doctypes={plan.get('doctypes')!r}")
    log.debug(f"SEARCH PLAN REASONING | {plan.get('reasoning')}")
    return plan


# ── Step 2: Indian Kanoon Search API ──────────────────────────────────────────
def search_indian_kanoon(form_input: str, doctypes: str = "", pagenum: int = 0, max_results: int = 5) -> list:
    """
    Search IK sorted by most recent.
    Returns (docs_for_context, recent_cases) where:
      - docs_for_context : up to max_results docs for RAG context
      - recent_cases     : top 2 most-recent docs as clean user-facing references
    """
    # Always sort by most recent so top results = latest judgments
    query = form_input + " sortby:mostrecent"
    if doctypes:
        query += f" doctypes:{doctypes}"

    t0   = time.time()
    resp = requests.post(f"{IK_BASE}/search/", headers=IK_HEADERS, data={"formInput": query, "pagenum": pagenum}, timeout=20)
    resp.raise_for_status()
    data    = resp.json()
    elapsed = time.time() - t0

    all_docs         = data.get("docs", [])
    docs_for_context = all_docs[:max_results]

    log.info(f"IK SEARCH | query={query!r}  found={data.get('found','?')}  returned={len(docs_for_context)}  time={elapsed:.2f}s")
    for i, d in enumerate(docs_for_context, 1):
        log.debug(f"  DOC {i} | tid={d.get('tid')}  source={d.get('docsource')}  size={d.get('docsize')}  title={d.get('title')!r}")

    return docs_for_context


# ── Step 3: Fetch doc fragments ───────────────────────────────────────────────
MAX_CHARS_PER_DOC = 3000

def fetch_doc_text(tid: int, form_input: str) -> str:
    """Fetch highlighted fragment; fall back to full doc."""
    import re

    try:
        resp = requests.post(
            f"{IK_BASE}/docfragment/{tid}/",
            headers=IK_HEADERS,
            data={"formInput": form_input},
            timeout=20,
        )
        resp.raise_for_status()
        headline = resp.json().get("headline", "")
        if headline:
            log.debug(f"  FRAGMENT fetched | tid={tid}  chars={len(headline)}")
            return headline[:MAX_CHARS_PER_DOC]
    except Exception as e:
        log.warning(f"  FRAGMENT failed | tid={tid}  err={e}")

    try:
        resp = requests.post(f"{IK_BASE}/doc/{tid}/", headers=IK_HEADERS, timeout=20)
        resp.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", resp.json().get("doc", ""))
        text = re.sub(r"\s+", " ", text).strip()
        log.debug(f"  FULL DOC fetched | tid={tid}  chars={len(text)}")
        return text[:MAX_CHARS_PER_DOC]
    except Exception as e:
        log.warning(f"  FULL DOC failed  | tid={tid}  err={e}")
        return ""


def build_context(docs: list, form_input: str) -> tuple:
    """Fetch text for each doc; return context string + citations metadata."""
    context_parts = []
    citations     = []

    for doc in docs:
        tid   = doc.get("tid")
        title = doc.get("title", "Unknown")
        src   = doc.get("docsource", "")
        url   = f"https://indiankanoon.org/doc/{tid}/"

        text = fetch_doc_text(tid, form_input)
        if not text:
            log.warning(f"  SKIPPED (no text) | tid={tid}  title={title!r}")
            continue

        context_parts.append(f"### {title} [{src}]\n{text}")
        citations.append({"title": title, "source": src, "url": url, "tid": tid})
        log.info(f"  CITATION | {title!r}  {url}")

    return "\n\n".join(context_parts), citations


# ── Step 4: Claude → user-friendly answer ────────────────────────────────────
ANSWER_SYNTHESISER_PROMPT = """
You are VaadAI, a legal information assistant for India helping ordinary people understand their rights.

You will receive a user's legal question and verified excerpts from Indian court judgments and statutes.

LENGTH RULES (strict):
- Your entire answer must be 6–10 lines maximum. No exceptions.
- One direct opening sentence answering the question. 
- Then 3–5 bullet points covering the key points. Each bullet: one line only.
- Then the disclaimer line. That's it.

CONTENT RULES:
- Plain, simple language — as if texting a friend. Zero jargon.
- If the answer comes directly from a specific law, mention only its short name and section number (e.g. "under Section 125 CrPC" or "under Rule 4 of the Maharashtra Rent Control Act"). Nothing more.
- If the law has been renamed (e.g. CrPC → BNSS), add it in brackets: "Section 482 CrPC (now Section 528 BNSS)".
- NEVER include case names, citation numbers, document IDs, or URLs.
- If context is insufficient, say so in one line and name the type of lawyer to consult.

End with exactly this line:
⚠️ This is legal information, not legal advice. Please consult a lawyer for your situation.
"""

def synthesise_answer(user_question: str, context: str, history: list = None) -> str:
    """
    Ask Claude to produce a plain-language answer.
    history: list of {"role": "user"|"assistant", "content": str} for follow-up awareness.
    """
    t0 = time.time()

    # Build messages: prior turns first, then current question with fresh IK context
    messages = []
    for turn in (history or []):
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Always append the current question with its retrieved context
    messages.append({
        "role": "user",
        "content": f"USER QUESTION:\n{user_question}\n\nINDIAN KANOON CONTEXT:\n{context}",
    })

    resp = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=ANSWER_SYNTHESISER_PROMPT,
        messages=messages,
    )
    elapsed = time.time() - t0
    answer  = resp.content[0].text.strip()
    log.info(f"SYNTHESIS | tokens_in={resp.usage.input_tokens}  tokens_out={resp.usage.output_tokens}  time={elapsed:.2f}s")
    return answer


# ── Step 4b: Fetch top 2 relevant cases via IK API (authenticated, never blocked) ──
def fetch_top_cases_from_ik_web(user_question: str) -> tuple:
    """
    Use the authenticated IK API to search for the most relevant recent cases.
    Falls back gracefully to an empty list — never crashes the pipeline.
    Returns (cases, search_url) where search_url is the equivalent public URL
    users can open in a browser to see more cases.
    """
    from urllib.parse import quote_plus

    # Step A: Claude extracts tight keywords
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
    keywords  = kw_resp.content[0].text.strip()
    log.info(f"CASE KEYWORDS | {keywords!r}")

    # Step B: Build public browse URL (for display only — not fetched)
    search_url = "https://indiankanoon.org/search/?formInput=" + quote_plus(keywords + " sortby:mostrecent")

    # Step C: Use authenticated API with sortby:mostrecent + doctypes:judgments
    query = keywords + " sortby:mostrecent doctypes:judgments"
    try:
        resp = requests.post(
            f"{IK_BASE}/search/",
            headers=IK_HEADERS,
            data={"formInput": query, "pagenum": 0},
            timeout=20,
        )
        resp.raise_for_status()
        docs = resp.json().get("docs", [])
        log.info(f"CASE API SEARCH | query={query!r}  found={len(docs)}")
    except Exception as e:
        log.warning(f"CASE API SEARCH failed: {e}")
        return [], search_url

    # Step D: Return top 2 distinct cases (deduplicate by title similarity)
    def title_similarity(a: str, b: str) -> float:
        """Simple character-level similarity — no external libs needed."""
        a, b = a.lower().strip(), b.lower().strip()
        if a == b:
            return 1.0
        # Count matching chars via longest common subsequence approximation:
        # use set of bigrams for speed (good enough for 97-98% threshold)
        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s) - 1))
        bg_a, bg_b = bigrams(a), bigrams(b)
        if not bg_a or not bg_b:
            return 0.0
        return 2 * len(bg_a & bg_b) / (len(bg_a) + len(bg_b))

    cases    = []
    for d in docs:                          # iterate ALL returned docs, not just [:2]
        tid   = d.get("tid")
        title = d.get("title", "Unknown")
        src   = d.get("docsource", "")
        url   = f"https://indiankanoon.org/doc/{tid}/"

        # Check this candidate isn't too similar to any already-picked case
        is_duplicate = any(title_similarity(title, c["title"]) >= 0.97 for c in cases)
        if is_duplicate:
            log.debug(f"  SKIPPED (duplicate) | {title!r}")
            continue

        cases.append({"title": title, "source": src, "url": url})
        log.debug(f"  CASE | {title!r}  {src}  {url}")

        if len(cases) == 2:
            break

    log.info(f"CASES SELECTED | {[c['title'] for c in cases]}")
    return cases, search_url


# ── Orchestrator ──────────────────────────────────────────────────────────────
def rag_query(user_question: str, history: list = None) -> str:
    """
    Full pipeline → returns only the plain-language answer string.
    history: list of {"role": "user"|"assistant", "content": str} from the frontend.
    All technical details (search plan, citations, timing) go to vaad_ai.log.
    """
    run_id = datetime.now().strftime("%H%M%S")
    log_section(f"RUN {run_id} | {user_question}")

    # 1. Plan search
    search_plan = plan_search(user_question)
    form_input  = search_plan["formInput"]
    doctypes    = search_plan.get("doctypes", "")

    # 2a. Search Indian Kanoon for RAG context (broader query)
    docs = search_indian_kanoon(form_input, doctypes, max_results=10)
    if not docs:
        log.warning("No documents returned from IK search.")
        return (
            "I wasn't able to find relevant legal information for your question right now. "
            "Please try rephrasing, or consult a local lawyer directly.\n\n"
            "⚠️ This is legal information, not legal advice. For your specific situation, please consult a lawyer."
        )



    # 3. Build context
    context, citations = build_context(docs, form_input)
    if not context.strip():
        log.warning("Context empty after fetching all docs.")
        context = "No usable context could be retrieved."

    log.info(f"CONTEXT BUILT | docs_used={len(citations)}  total_chars={len(context)}")

    # 4. Synthesise answer
    answer = synthesise_answer(user_question, context, history=history)

    # 5. Fetch top 2 relevant cases from IK public search HTML
    relevant_cases, ik_search_url = fetch_top_cases_from_ik_web(user_question)

    if relevant_cases:
        cases_block = "\n\n📌 Recent related cases:"
        for c in relevant_cases:
            cases_block += f"\n  • {c['title']} — {c['source']}\n    {c['url']}"
        cases_block += f"\n\n🔗 Verified via: {ik_search_url}"
        answer += cases_block

    log.info(f"RUN {run_id} COMPLETE")
    return answer


# ── Test Queries ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    "Can FIR be quashed under Section 482 CrPC? What are the grounds courts have accepted?",
    "Landlord is trying to evict my client without 30-day notice. Which Rent Control Act sections apply in Maharashtra?",
    "How is maintenance calculated under Section 125 CrPC? What factors does the court consider?",
]

if __name__ == "__main__":
    print("🚀 VaadAI  |  Technical logs → vaad_ai.log\n" + "═" * 60)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\nQuestion {i}: {query}\n")
        answer = rag_query(query)
        print(answer)
        print("\n" + "─" * 60)

    print("\n✅ Done. Check vaad_ai.log for full technical details.")


























# """
# VaadAI – Indian Kanoon RAG Pipeline
# Flow: User Query → Claude (generates search query) → Indian Kanoon API
#       → Claude (synthesises answer) → User-friendly response

# User sees  : plain-language answer only
# Logged to  : vaad_ai.log — search plan, citations, doc metadata, timing
# """

# import os
# import json
# import logging
# import time
# import requests
# from datetime import datetime
# from dotenv import load_dotenv
# import anthropic

# load_dotenv()

# # ── Logger (all technical details go here, not to stdout) ─────────────────────
# LOG_FILE = "vaad_ai.log"
# logging.basicConfig(
#     filename=LOG_FILE,
#     level=logging.DEBUG,
#     format="%(asctime)s  %(levelname)-8s  %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# log = logging.getLogger("vaadai")

# def log_section(title: str):
#     log.debug("─" * 60)
#     log.debug(f"  {title}")
#     log.debug("─" * 60)


# # ── Clients ────────────────────────────────────────────────────────────────────
# claude   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# IK_TOKEN = os.getenv("INDIAN_KANOON_API_KEY")
# if not IK_TOKEN:
#     raise EnvironmentError("INDIAN_KANOON_API_KEY is missing from your .env file")

# IK_BASE    = "https://api.indiankanoon.org"
# IK_HEADERS = {
#     "Authorization": f"Token {IK_TOKEN}",
#     "Accept": "application/json",
# }


# # ── Step 1: Claude → optimised IK search query ────────────────────────────────
# SEARCH_PLANNER_PROMPT = """
# You are a legal search specialist for Indian law.
# Given a user's legal question, output a JSON object (and NOTHING else) with:

# {
#   "formInput": "<concise Indian Kanoon search query>",
#   "doctypes":  "<one of: supremecourt | highcourts | judgments | laws | tribunals | (blank for all)>",
#   "reasoning": "<one sentence on why you chose this query>"
# }

# Rules:
# - Keep formInput short (3-8 words), keyword-rich, and suited for Indian Kanoon full-text search.
# - Use ANDD / ORR / NOTT operators where useful (they are case-sensitive).
# - Pick doctypes that best fits the question; leave blank if unsure.
# - Output only valid JSON — no markdown fences, no preamble.
# """

# def plan_search(user_question: str) -> dict:
#     """Ask Claude to craft the best Indian Kanoon search query."""
#     resp = claude.messages.create(
#         model="claude-sonnet-4-20250514",
#         max_tokens=256,
#         system=SEARCH_PLANNER_PROMPT,
#         messages=[{"role": "user", "content": user_question}],
#     )
#     raw  = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
#     plan = json.loads(raw)
#     log.info(f"SEARCH PLAN | formInput={plan['formInput']!r}  doctypes={plan.get('doctypes')!r}")
#     log.debug(f"SEARCH PLAN REASONING | {plan.get('reasoning')}")
#     return plan


# # ── Step 2: Indian Kanoon Search API ──────────────────────────────────────────
# def search_indian_kanoon(form_input: str, doctypes: str = "", pagenum: int = 0, max_results: int = 5) -> list:
#     """
#     Search IK sorted by most recent.
#     Returns (docs_for_context, recent_cases) where:
#       - docs_for_context : up to max_results docs for RAG context
#       - recent_cases     : top 2 most-recent docs as clean user-facing references
#     """
#     # Always sort by most recent so top results = latest judgments
#     query = form_input + " sortby:mostrecent"
#     if doctypes:
#         query += f" doctypes:{doctypes}"

#     t0   = time.time()
#     resp = requests.post(f"{IK_BASE}/search/", headers=IK_HEADERS, data={"formInput": query, "pagenum": pagenum}, timeout=20)
#     resp.raise_for_status()
#     data    = resp.json()
#     elapsed = time.time() - t0

#     all_docs         = data.get("docs", [])
#     docs_for_context = all_docs[:max_results]

#     log.info(f"IK SEARCH | query={query!r}  found={data.get('found','?')}  returned={len(docs_for_context)}  time={elapsed:.2f}s")
#     for i, d in enumerate(docs_for_context, 1):
#         log.debug(f"  DOC {i} | tid={d.get('tid')}  source={d.get('docsource')}  size={d.get('docsize')}  title={d.get('title')!r}")

#     return docs_for_context


# # ── Step 3: Fetch doc fragments ───────────────────────────────────────────────
# MAX_CHARS_PER_DOC = 3000

# def fetch_doc_text(tid: int, form_input: str) -> str:
#     """Fetch highlighted fragment; fall back to full doc."""
#     import re

#     try:
#         resp = requests.post(
#             f"{IK_BASE}/docfragment/{tid}/",
#             headers=IK_HEADERS,
#             data={"formInput": form_input},
#             timeout=20,
#         )
#         resp.raise_for_status()
#         headline = resp.json().get("headline", "")
#         if headline:
#             log.debug(f"  FRAGMENT fetched | tid={tid}  chars={len(headline)}")
#             return headline[:MAX_CHARS_PER_DOC]
#     except Exception as e:
#         log.warning(f"  FRAGMENT failed | tid={tid}  err={e}")

#     try:
#         resp = requests.post(f"{IK_BASE}/doc/{tid}/", headers=IK_HEADERS, timeout=20)
#         resp.raise_for_status()
#         text = re.sub(r"<[^>]+>", " ", resp.json().get("doc", ""))
#         text = re.sub(r"\s+", " ", text).strip()
#         log.debug(f"  FULL DOC fetched | tid={tid}  chars={len(text)}")
#         return text[:MAX_CHARS_PER_DOC]
#     except Exception as e:
#         log.warning(f"  FULL DOC failed  | tid={tid}  err={e}")
#         return ""


# def build_context(docs: list, form_input: str) -> tuple:
#     """Fetch text for each doc; return context string + citations metadata."""
#     context_parts = []
#     citations     = []

#     for doc in docs:
#         tid   = doc.get("tid")
#         title = doc.get("title", "Unknown")
#         src   = doc.get("docsource", "")
#         url   = f"https://indiankanoon.org/doc/{tid}/"

#         text = fetch_doc_text(tid, form_input)
#         if not text:
#             log.warning(f"  SKIPPED (no text) | tid={tid}  title={title!r}")
#             continue

#         context_parts.append(f"### {title} [{src}]\n{text}")
#         citations.append({"title": title, "source": src, "url": url, "tid": tid})
#         log.info(f"  CITATION | {title!r}  {url}")

#     return "\n\n".join(context_parts), citations


# # ── Step 4: Claude → user-friendly answer ────────────────────────────────────
# ANSWER_SYNTHESISER_PROMPT = """
# You are VaadAI, a legal information assistant for India helping ordinary people understand their rights.

# You will receive a user's legal question and verified excerpts from Indian court judgments and statutes.

# LENGTH RULES (strict):
# - Your entire answer must be 6–10 lines maximum. No exceptions.
# - One direct opening sentence answering the question. 
# - Then 3–5 bullet points covering the key points. Each bullet: one line only.
# - Then the disclaimer line. That's it.

# CONTENT RULES:
# - Plain, simple language — as if texting a friend. Zero jargon.
# - If the answer comes directly from a specific law, mention only its short name and section number (e.g. "under Section 125 CrPC" or "under Rule 4 of the Maharashtra Rent Control Act"). Nothing more.
# - If the law has been renamed (e.g. CrPC → BNSS), add it in brackets: "Section 482 CrPC (now Section 528 BNSS)".
# - NEVER include case names, citation numbers, document IDs, or URLs.
# - If context is insufficient, say so in one line and name the type of lawyer to consult.

# End with exactly this line:
# ⚠️ This is legal information, not legal advice. Please consult a lawyer for your situation.
# """

# def synthesise_answer(user_question: str, context: str) -> str:
#     """Ask Claude to produce a plain-language answer."""
#     t0   = time.time()
#     resp = claude.messages.create(
#         model="claude-sonnet-4-20250514",
#         max_tokens=400,
#         system=ANSWER_SYNTHESISER_PROMPT,
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"USER QUESTION:\n{user_question}\n\nINDIAN KANOON CONTEXT:\n{context}",
#             }
#         ],
#     )
#     elapsed = time.time() - t0
#     answer  = resp.content[0].text.strip()
#     log.info(f"SYNTHESIS | tokens_in={resp.usage.input_tokens}  tokens_out={resp.usage.output_tokens}  time={elapsed:.2f}s")
#     return answer


# # ── Step 4b: Fetch top 2 relevant cases via IK API (authenticated, never blocked) ──
# def fetch_top_cases_from_ik_web(user_question: str) -> tuple:
#     """
#     Use the authenticated IK API to search for the most relevant recent cases.
#     Falls back gracefully to an empty list — never crashes the pipeline.
#     Returns (cases, search_url) where search_url is the equivalent public URL
#     users can open in a browser to see more cases.
#     """
#     from urllib.parse import quote_plus

#     # Step A: Claude extracts tight keywords
#     kw_resp = claude.messages.create(
#         model="claude-sonnet-4-20250514",
#         max_tokens=32,
#         system=(
#             "Extract a 3-6 word keyword phrase describing the SPECIFIC legal issue "
#             "(e.g. 'FIR quash Section 482 CrPC', 'eviction non-payment rent', "
#             "'maintenance Section 125 CrPC'). Output ONLY the phrase, nothing else."
#         ),
#         messages=[{"role": "user", "content": user_question}],
#     )
#     keywords  = kw_resp.content[0].text.strip()
#     log.info(f"CASE KEYWORDS | {keywords!r}")

#     # Step B: Build public browse URL (for display only — not fetched)
#     search_url = "https://indiankanoon.org/search/?formInput=" + quote_plus(keywords + " sortby:mostrecent")

#     # Step C: Use authenticated API with sortby:mostrecent + doctypes:judgments
#     query = keywords + " sortby:mostrecent doctypes:judgments"
#     try:
#         resp = requests.post(
#             f"{IK_BASE}/search/",
#             headers=IK_HEADERS,
#             data={"formInput": query, "pagenum": 0},
#             timeout=20,
#         )
#         resp.raise_for_status()
#         docs = resp.json().get("docs", [])
#         log.info(f"CASE API SEARCH | query={query!r}  found={len(docs)}")
#     except Exception as e:
#         log.warning(f"CASE API SEARCH failed: {e}")
#         return [], search_url

#     # Step D: Return top 2 distinct cases (deduplicate by title similarity)
#     def title_similarity(a: str, b: str) -> float:
#         """Simple character-level similarity — no external libs needed."""
#         a, b = a.lower().strip(), b.lower().strip()
#         if a == b:
#             return 1.0
#         # Count matching chars via longest common subsequence approximation:
#         # use set of bigrams for speed (good enough for 97-98% threshold)
#         def bigrams(s):
#             return set(s[i:i+2] for i in range(len(s) - 1))
#         bg_a, bg_b = bigrams(a), bigrams(b)
#         if not bg_a or not bg_b:
#             return 0.0
#         return 2 * len(bg_a & bg_b) / (len(bg_a) + len(bg_b))

#     cases    = []
#     for d in docs:                          # iterate ALL returned docs, not just [:2]
#         tid   = d.get("tid")
#         title = d.get("title", "Unknown")
#         src   = d.get("docsource", "")
#         url   = f"https://indiankanoon.org/doc/{tid}/"

#         # Check this candidate isn't too similar to any already-picked case
#         is_duplicate = any(title_similarity(title, c["title"]) >= 0.97 for c in cases)
#         if is_duplicate:
#             log.debug(f"  SKIPPED (duplicate) | {title!r}")
#             continue

#         cases.append({"title": title, "source": src, "url": url})
#         log.debug(f"  CASE | {title!r}  {src}  {url}")

#         if len(cases) == 2:
#             break

#     log.info(f"CASES SELECTED | {[c['title'] for c in cases]}")
#     return cases, search_url


# # ── Orchestrator ──────────────────────────────────────────────────────────────
# def rag_query(user_question: str) -> str:
#     """
#     Full pipeline → returns only the plain-language answer string.
#     All technical details (search plan, citations, timing) go to vaad_ai.log.
#     """
#     run_id = datetime.now().strftime("%H%M%S")
#     log_section(f"RUN {run_id} | {user_question}")

#     # 1. Plan search
#     search_plan = plan_search(user_question)
#     form_input  = search_plan["formInput"]
#     doctypes    = search_plan.get("doctypes", "")

#     # 2a. Search Indian Kanoon for RAG context (broader query)
#     docs = search_indian_kanoon(form_input, doctypes, max_results=10)
#     if not docs:
#         log.warning("No documents returned from IK search.")
#         return (
#             "I wasn't able to find relevant legal information for your question right now. "
#             "Please try rephrasing, or consult a local lawyer directly.\n\n"
#             "⚠️ This is legal information, not legal advice. For your specific situation, please consult a lawyer."
#         )



#     # 3. Build context
#     context, citations = build_context(docs, form_input)
#     if not context.strip():
#         log.warning("Context empty after fetching all docs.")
#         context = "No usable context could be retrieved."

#     log.info(f"CONTEXT BUILT | docs_used={len(citations)}  total_chars={len(context)}")

#     # 4. Synthesise answer
#     answer = synthesise_answer(user_question, context)

#     # 5. Fetch top 2 relevant cases from IK public search HTML
#     relevant_cases, ik_search_url = fetch_top_cases_from_ik_web(user_question)

#     if relevant_cases:
#         cases_block = "\n\n📌 Recent related cases:"
#         for c in relevant_cases:
#             cases_block += f"\n  • {c['title']} — {c['source']}\n    {c['url']}"
#         cases_block += f"\n\n🔗 Verified via: {ik_search_url}"
#         answer += cases_block

#     log.info(f"RUN {run_id} COMPLETE")
#     return answer


# # ── Test Queries ──────────────────────────────────────────────────────────────
# TEST_QUERIES = [
#     "Can FIR be quashed under Section 482 CrPC? What are the grounds courts have accepted?",
#     "Landlord is trying to evict my client without 30-day notice. Which Rent Control Act sections apply in Maharashtra?",
#     "How is maintenance calculated under Section 125 CrPC? What factors does the court consider?",
# ]

# if __name__ == "__main__":
#     print("🚀 VaadAI  |  Technical logs → vaad_ai.log\n" + "═" * 60)

#     for i, query in enumerate(TEST_QUERIES, 1):
#         print(f"\nQuestion {i}: {query}\n")
#         answer = rag_query(query)
#         print(answer)
#         print("\n" + "─" * 60)

#     print("\n✅ Done. Check vaad_ai.log for full technical details.")

