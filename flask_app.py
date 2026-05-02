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


# In-memory conversation history per user
from collections import defaultdict
conversation_history = defaultdict(list)
MAX_MESSAGES = 8  # 8 user messages per session


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

        if "messages" not in value:
            print("[WA] Ignoring non-message event", flush=True)
            return jsonify({"status": "ok"}), 200

        message    = value["messages"][0]
        user_phone = message["from"]
        msg_type   = message.get("type")

        print(f"[WA] Received message type: {msg_type} from {user_phone}", flush=True)

        if msg_type == "text":
            user_text = message["text"]["body"]
        elif msg_type == "interactive":
            reply = message["interactive"]["list_reply"]
            user_text = reply.get("description") or reply.get("title")
        else:
            print(f"[WA] Ignoring message type: {msg_type}", flush=True)
            return jsonify({"status": "ok"}), 200

        print(f"[WA] User asked: {user_text}", flush=True)


        
        # ── Check message limit ───────────────────────────────────────────────
        user_history = conversation_history[user_phone]
        user_message_count = sum(1 for m in user_history if m["role"] == "user")

        if user_message_count >= MAX_MESSAGES:
            # Clear history and notify user
            conversation_history[user_phone] = []
            limit_msg = (
                "You have reached the limit of 8 follow-up questions. "
                "To continue, upgrade your plan or ask a new question."
            )
            send_whatsapp_message(user_phone, limit_msg)
            print(f"[WA] Message limit reached for {user_phone}, history cleared.", flush=True)
            return jsonify({"status": "ok"}), 200

        # ── Run RAG pipeline with history ─────────────────────────────────────
        result = rag_query(user_text, history=user_history)

        answer_raw = result.get("answer", "")
        if isinstance(answer_raw, list):
            answer = " ".join(str(a) for a in answer_raw)
        else:
            answer = str(answer_raw) if answer_raw else "Sorry, I couldn't find an answer."

        citations   = result.get("citations", [])
        suggestions = result.get("suggestions", [])

        print(f"[WA] RAG answered. Citations: {len(citations)}, Suggestions: {len(suggestions)}", flush=True)

        # ── Update history ────────────────────────────────────────────────────
        conversation_history[user_phone].append({"role": "user",      "content": user_text})
        conversation_history[user_phone].append({"role": "assistant", "content": answer})

        # Trim to last MAX_HISTORY_TURNS exchanges
        if len(conversation_history[user_phone]) > MAX_HISTORY_TURNS * 2:
            conversation_history[user_phone] = conversation_history[user_phone][-(MAX_HISTORY_TURNS * 2):]

        # ── Build response text ───────────────────────────────────────────────
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

        # ── Send response ─────────────────────────────────────────────────────
        r1 = send_whatsapp_message(user_phone, main_text)
        print(f"[WA] Plain text response: {r1.status_code}", flush=True)

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
        if len(text) <= limit:
            return text
        truncated = text[:limit].rsplit(" ", 1)[0]
        return truncated.rstrip(".,?") + "…"
    
    rows = [
        {
            "id": f"suggestion_{i}",
            "title": truncate_title(s),  # WhatsApp requires this, keep it short
            "description": s             # ✅ Full question always stored here
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
