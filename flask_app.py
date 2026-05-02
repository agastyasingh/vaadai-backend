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





# Adding whatsapp routes

VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "vaadaiVerificationToken2001")
WA_TOKEN = os.environ.get("WA_TOKEN")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID")

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Forbidden", 403

@app.route("/webhook", methods=["POST"])
def receive_message():
    data = request.get_json()
    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        user_phone = message["from"]
        user_text = message["text"]["body"]

        result = rag_query(user_text)           # ✅ your actual RAG function
        rag_response = result.get("answer", "Sorry, I couldn't find an answer.")

        send_whatsapp_message(user_phone, rag_response)
    except (KeyError, IndexError):
        pass
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
    requests.post(url, headers=headers, json=payload)


if __name__ == "__main__":
    app.run(debug=False)
