import sys
import os
import threading
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from claude_rag_test import rag_query, generate_follow_up_questions
import session_store as ss

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "VaadAI is running"})


# ── Web JSON endpoints ────────────────────────────────────────────────────────

def _clean_history(history):
    return [
        {"role": t["role"], "content": t["content"]}
        for t in (history or [])
        if isinstance(t, dict)
        and t.get("role") in ("user", "assistant")
        and isinstance(t.get("content"), str)
    ]


def _session_key_from_request(data: dict) -> str:
    """
    Web clients pass `session_id` (preferred) or `user_id`. We key the SQLite
    store by that string. This keeps the WhatsApp side (keyed on phone number)
    and the web side (keyed on the client-supplied id) on the same machinery.
    Falls back to remote_addr only if the client sends nothing — that's
    deliberately weak so abuse is obvious.
    """
    sid = (data.get("session_id") or data.get("user_id") or "").strip()
    if sid:
        return f"web:{sid}"
    return f"ip:{request.remote_addr or 'unknown'}"


def _rag_json_response(result: dict, session_id: str, question_count: int):
    sug = result.get("suggestions", [])
    return jsonify({
        "answer": result.get("answer", ""),
        "suggestions": sug,
        "recommendations": sug,
        "citations": result.get("citations") or [],
        "disclaimer": result.get("disclaimer"),
        "more_cases_url": result.get("more_cases_url"),
        "session_id": session_id,
        "question_count": question_count,
        "questions_remaining": max(0, ss.MAX_QUESTIONS_PER_SESSION - question_count),
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    is_suggested = bool(data.get("is_suggested", False))
    user_key = _session_key_from_request(data)

    sess = ss.get_or_rotate_session(user_key)

    if sess["question_count"] >= ss.MAX_QUESTIONS_PER_SESSION:
        return jsonify({
            "error": "limit_reached",
            "message": (
                f"You have reached the limit of {ss.MAX_QUESTIONS_PER_SESSION} "
                "questions for this session. Please start a new session."
            ),
            "session_id": sess["session_id"],
        }), 429

    # Suggested follow-ups get conversation history; standalone questions don't.
    history = sess["history"] if is_suggested else []
    if "history" in data:  # explicit override from older clients
        history = _clean_history(data["history"]) if is_suggested else []

    try:
        result = rag_query(question, history=history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    answer = result.get("answer", "") or ""
    ss.append_history(user_key, question, answer)
    new_count = ss.increment_question_count(user_key)
    if result.get("suggestions"):
        ss.store_suggestions(user_key, result["suggestions"])

    return _rag_json_response(result, sess["session_id"], new_count)


@app.route("/recommendations", methods=["POST"])
def recommendations():
    """
    Fast path: question + answer (and optional context) -> follow-ups only.
    Fallback: if answer is omitted, runs full rag_query.
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    extra_ctx = (data.get("context") or "").strip()
    is_suggested = bool(data.get("is_suggested", False))

    if not question:
        return jsonify({"error": "No question provided"}), 400

    user_key = _session_key_from_request(data)
    sess = ss.get_or_rotate_session(user_key)
    history = sess["history"] if is_suggested else []
    if "history" in data:
        history = _clean_history(data["history"]) if is_suggested else []

    if answer:
        try:
            sug = generate_follow_up_questions(
                question, answer, context=extra_ctx, history=history
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        if sug:
            ss.store_suggestions(user_key, sug)
        return jsonify({
            "recommendations": sug,
            "suggestions": sug,
            "answer": answer,
            "session_id": sess["session_id"],
        })

    # Fallback path also counts as a question, so it must respect the cap.
    if sess["question_count"] >= ss.MAX_QUESTIONS_PER_SESSION:
        return jsonify({
            "error": "limit_reached",
            "session_id": sess["session_id"],
        }), 429

    try:
        result = rag_query(question, history=history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    ss.append_history(user_key, question, result.get("answer", "") or "")
    new_count = ss.increment_question_count(user_key)
    if result.get("suggestions"):
        ss.store_suggestions(user_key, result["suggestions"])

    return jsonify({
        "recommendations": result.get("suggestions", []),
        "suggestions": result.get("suggestions", []),
        "answer": result.get("answer", ""),
        "citations": result.get("citations") or [],
        "disclaimer": result.get("disclaimer"),
        "more_cases_url": result.get("more_cases_url"),
        "session_id": sess["session_id"],
        "question_count": new_count,
        "questions_remaining": max(0, ss.MAX_QUESTIONS_PER_SESSION - new_count),
    })


# ── WhatsApp ──────────────────────────────────────────────────────────────────

VERIFY_TOKEN    = os.environ.get("VERIFY_TOKEN", "vaadaiVerificationToken2001")
WA_TOKEN        = os.environ.get("WA_TOKEN")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID")


@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
def receive_message():
    """
    Webhook contract: parse, enqueue, return 200 fast. The actual RAG work
    happens in a background worker so WhatsApp doesn't time out and re-deliver
    (which used to cause "answers showing up 3-4 hours later").
    """
    data = request.get_json(silent=True) or {}
    try:
        value = data["entry"][0]["changes"][0]["value"]

        if "messages" not in value:
            print("[WA] Ignoring non-message event", flush=True)
            return jsonify({"status": "ok"}), 200

        message    = value["messages"][0]
        user_phone = message["from"]
        msg_type   = message.get("type")

        print(f"[WA] Received message type: {msg_type} from {user_phone}", flush=True)

        if msg_type == "text":
            user_text    = message["text"]["body"]
            is_suggested = False
        elif msg_type == "interactive":
            reply        = message["interactive"]["list_reply"]
            reply_id     = reply.get("id", "")
            stored       = ss.lookup_suggestion(user_phone, reply_id)
            user_text    = stored or reply.get("description") or reply.get("title") or ""
            is_suggested = True
            print(f"[WA] Tapped suggestion id={reply_id} -> {user_text}", flush=True)
        else:
            print(f"[WA] Ignoring message type: {msg_type}", flush=True)
            return jsonify({"status": "ok"}), 200

        if not user_text.strip():
            return jsonify({"status": "ok"}), 200

        sess = ss.get_or_rotate_session(user_phone)

        if sess["question_count"] >= ss.MAX_QUESTIONS_PER_SESSION:
            send_whatsapp_message(
                user_phone,
                f"You have reached the limit of {ss.MAX_QUESTIONS_PER_SESSION} "
                "questions for this session. To continue, please start a new "
                "conversation in 10 minutes or upgrade your plan."
            )
            print(f"[WA] Session cap hit for {user_phone}", flush=True)
            return jsonify({"status": "ok"}), 200

        ss.enqueue_message(user_phone, sess["session_id"], user_text, is_suggested)
        print(f"[WA] Enqueued (is_suggested={is_suggested}) for {user_phone}", flush=True)

        # Kick off a worker. If another worker already holds the lock for this
        # user, this thread will exit immediately — the in-flight worker will
        # drain the queue.
        threading.Thread(
            target=_drain_user_queue,
            args=(user_phone,),
            daemon=True,
        ).start()

    except Exception as e:
        print(f"[WA ERROR] {type(e).__name__}: {e}", flush=True)

    return jsonify({"status": "ok"}), 200


def _drain_user_queue(user_phone: str) -> None:
    """
    Background worker: pull messages for `user_phone` one at a time, in order,
    until the queue is empty. Only one worker per user runs at a time (enforced
    via the SQLite-backed processing lock). Stale items (>10min old) are
    discarded inside pop_next_message.
    """
    if not ss.try_acquire_processing_lock(user_phone):
        return

    try:
        while True:
            item = ss.pop_next_message(user_phone)
            if item is None:
                # Atomically check-and-release; if a new message arrived in the
                # window between our pop and this check, keep going.
                if ss.release_lock_if_queue_empty(user_phone):
                    return
                continue

            try:
                _process_one(user_phone, item)
            except Exception as e:
                print(
                    f"[WA WORKER ERROR] {user_phone}: {type(e).__name__}: {e}",
                    flush=True,
                )
                try:
                    send_whatsapp_message(
                        user_phone,
                        "Sorry, something went wrong while answering that. "
                        "Please try again in a moment."
                    )
                except Exception:
                    pass
    finally:
        # Idempotent safety net for crashes — release_lock_if_queue_empty
        # already handled the normal path.
        ss.release_processing_lock(user_phone)


def _process_one(user_phone: str, item: dict) -> None:
    user_text    = item["text"]
    is_suggested = item["is_suggested"]
    enqueued_sid = item["session_id"]

    # Re-check the session at processing time. The session may have rotated
    # while this item sat in the queue (e.g. user idled out). If the message
    # belongs to an old session, drop it — we don't want to answer stale chips
    # against a fresh session.
    current = ss.get_session_snapshot(user_phone)
    if current is None or current["session_id"] != enqueued_sid:
        print(f"[WA] Dropping stale-session item for {user_phone}", flush=True)
        return

    if current["question_count"] >= ss.MAX_QUESTIONS_PER_SESSION:
        send_whatsapp_message(
            user_phone,
            f"You have reached the limit of {ss.MAX_QUESTIONS_PER_SESSION} "
            "questions for this session."
        )
        return

    # Suggested taps get conversation history so follow-ups read coherently;
    # typed (unique) questions get a clean slate so unrelated topics don't
    # bleed into context.
    history = current["history"] if is_suggested else []

    print(
        f"[WA] Processing (is_suggested={is_suggested}) "
        f"sid={enqueued_sid[:8]} -> {user_text}",
        flush=True,
    )

    result = rag_query(user_text, history=history)
    answer_raw = result.get("answer", "")
    if isinstance(answer_raw, list):
        answer = " ".join(str(a) for a in answer_raw)
    else:
        answer = str(answer_raw) if answer_raw else "Sorry, I couldn't find an answer."

    citations   = result.get("citations", []) or []
    suggestions = result.get("suggestions", []) or []

    ss.append_history(user_phone, user_text, answer)
    ss.increment_question_count(user_phone)

    main_text = answer
    if citations:
        main_text += "\n\n📚 *References:*"
        for c in citations[:3]:
            title = c.get("title") or c.get("name") or "Source"
            url   = c.get("url") or c.get("link") or c.get("href") or ""
            if url:
                main_text += f"\n• {title}\n  🔗 {url}"
            else:
                main_text += f"\n• {title}"
    main_text += "\n\n⚠️ _This is legal information, not legal advice. Please consult a lawyer._"

    r1 = send_whatsapp_message(user_phone, main_text)
    print(f"[WA] Plain text response: {r1.status_code}", flush=True)

    if suggestions:
        ss.store_suggestions(user_phone, suggestions)
        follow_up_text = "💡 Based on your question, you may also want to ask:"
        r2 = send_whatsapp_interactive(user_phone, follow_up_text, suggestions[:10])
        print(f"[WA] Interactive message: {r2.status_code}", flush=True)


def send_whatsapp_message(to, text):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    return requests.post(url, headers=headers, json=payload)


def send_whatsapp_interactive(to, body_text, suggestions):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json",
    }

    def truncate_title(text, limit=24):
        if len(text) <= limit:
            return text
        truncated = text[:limit].rsplit(" ", 1)[0]
        return truncated.rstrip(".,?") + "…"

    def truncate_description(text, limit=72):
        if len(text) <= limit:
            return text
        truncated = text[:limit].rsplit(" ", 1)[0]
        return truncated.rstrip(".,?") + "…"

    rows = [
        {
            "id": f"suggestion_{i}",
            "title": truncate_title(s),
            "description": truncate_description(s),
        }
        for i, s in enumerate(suggestions)
    ]

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": body_text},
            "action": {
                "button": "💡 Related Questions",
                "sections": [{"title": "You can also ask:", "rows": rows}],
            },
        },
    }
    return requests.post(url, headers=headers, json=payload)


if __name__ == "__main__":
    app.run(debug=False)
