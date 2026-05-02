import sys
import os
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from claude_rag_test import rag_query, generate_follow_up_questions

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "VaadAI is running"})

def _parse_ask_body():
    """Returns (question, clean_history) or raises ValueError with message for 400."""
    data     = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    history  = data.get("history", [])

    if not question:
        raise ValueError("No question provided")

    clean_history = [
        {"role": t["role"], "content": t["content"]}
        for t in history
        if isinstance(t, dict)
        and t.get("role") in ("user", "assistant")
        and isinstance(t.get("content"), str)
    ]
    return question, clean_history


def _rag_json_response(result: dict):
    """Single shape for /ask: suggestions, recommendations, citations, disclaimer."""
    sug = result.get("suggestions", [])
    payload = {
        "answer": result.get("answer", ""),
        "suggestions": sug,
        "recommendations": sug,
        "citations": result.get("citations") or [],
        "disclaimer": result.get("disclaimer"),
        "more_cases_url": result.get("more_cases_url"),
    }
    return jsonify(payload)


@app.route("/ask", methods=["POST"])
def ask():
    try:
        question, clean_history = _parse_ask_body()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        result = rag_query(question, history=clean_history)
        return _rag_json_response(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _parse_history_only():
    data = request.get_json(silent=True) or {}
    history = data.get("history", [])
    return [
        {"role": t["role"], "content": t["content"]}
        for t in history
        if isinstance(t, dict)
        and t.get("role") in ("user", "assistant")
        and isinstance(t.get("content"), str)
    ]


@app.route("/recommendations", methods=["POST"])
def recommendations():
    """
    Fast path: POST JSON with question + answer (and optional context) to generate follow-ups only
    (one short Claude call). Use this when the UI fetches chips after the main /ask response.

    Fallback: if answer is omitted, runs full rag_query (slow; avoid if possible).
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    extra_ctx = (data.get("context") or "").strip()
    clean_history = _parse_history_only()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if answer:
        try:
            sug = generate_follow_up_questions(
                question, answer, context=extra_ctx, history=clean_history
            )
            return jsonify({
                "recommendations": sug,
                "suggestions": sug,
                "answer": answer,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    try:
        result = rag_query(question, history=clean_history)
        return jsonify({
            "recommendations": result.get("suggestions", []),
            "suggestions": result.get("suggestions", []),
            "answer": result.get("answer", ""),
            "citations": result.get("citations") or [],
            "disclaimer": result.get("disclaimer"),
            "more_cases_url": result.get("more_cases_url"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── WhatsApp ──────────────────────────────────────────────────────────────────

VERIFY_TOKEN   = os.environ.get("VERIFY_TOKEN", "vaadaiVerificationToken2001")
WA_TOKEN       = os.environ.get("WA_TOKEN")
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
    data = request.get_json()
    try:
        value = data["entry"][0]["changes"][0]["value"]

        # ── Ignore status updates (delivered, read, sent) ─────────────────────
        if "messages" not in value:
            print("[WA] Ignoring non-message event (status update etc.)", flush=True)
            return jsonify({"status": "ok"}), 200

        message    = value["messages"][0]
        user_phone = message["from"]
        msg_type   = message.get("type")

        print(f"[WA] Received message type: {msg_type} from {user_phone}", flush=True)

        if msg_type == "text":
            user_text = message["text"]["body"]
        elif msg_type == "interactive":
            user_text = message["interactive"]["list_reply"]["title"]
        else:
            print(f"[WA] Ignoring message type: {msg_type}", flush=True)
            return jsonify({"status": "ok"}), 200

        print(f"[WA] User asked: {user_text}", flush=True)

        result      = rag_query(user_text)
        answer      = result.get("answer", "Sorry, I couldn't find an answer.")
        citations   = result.get("citations", [])
        suggestions = result.get("suggestions", [])

        print(f"[WA] RAG answered. Citations: {len(citations)}, Suggestions: {len(suggestions)}", flush=True)

        # ── Build main answer message (plain text, no length limit) ───────────
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

        # ── Always send full answer as plain text first ───────────────────────
        r1 = send_whatsapp_message(user_phone, main_text)
        print(f"[WA] Plain text response: {r1.status_code}", flush=True)

        # ── Then send suggestions as a separate interactive message ───────────
        if suggestions:
            follow_up_text = "💡 Based on your question, you may also want to ask:"
            r2 = send_whatsapp_interactive(user_phone, follow_up_text, suggestions[:10])
            print(f"[WA] Interactive message: {r2.status_code} {r2.text}", flush=True)

    except Exception as e:
        print(f"[WA ERROR] {type(e).__name__}: {e}", flush=True)

    return jsonify({"status": "ok"}), 200




def send_whatsapp_message(to, text):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    return requests.post(url, headers=headers, json=payload)  # ← return added


def send_whatsapp_interactive(to, body_text, suggestions):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json"
    }

    def truncate_title(text, limit=24):
        """Truncate at word boundary, add ellipsis if needed."""
        if len(text) <= limit:
            return text
        truncated = text[:limit].rsplit(" ", 1)[0]
        return truncated.rstrip(".,?") + "…"

    rows = [
        {
            "id": f"suggestion_{i}",
            "title": truncate_title(s),  # ← clean word-boundary truncation
            "description": s             # ← always show full question here
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
                "sections": [{"title": "You can also ask:", "rows": rows}]
            }
        }
    }
    return requests.post(url, headers=headers, json=payload)



if __name__ == "__main__":
    app.run(debug=False)
