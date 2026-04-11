import sys
import os

sys.path.insert(0, '/home/agastyasingh927/mysite')

from flask import Flask, request, jsonify
from flask_cors import CORS
from claude_rag_test import rag_query

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
    """Single shape for /ask and /recommendations: suggestions + recommendations alias."""
    sug = result.get("suggestions", [])
    payload = {
        "answer": result.get("answer", ""),
        "suggestions": sug,
        "recommendations": sug,
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


@app.route("/recommendations", methods=["POST"])
def recommendations():
    """
    Same pipeline as /ask; use when the UI loads follow-ups in a second request.
    Returns only recommendations/suggestions (and answer for convenience).
    Prefer a single /ask call to avoid duplicate RAG work.
    """
    try:
        question, clean_history = _parse_ask_body()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        result = rag_query(question, history=clean_history)
        return jsonify({
            "recommendations": result.get("suggestions", []),
            "suggestions": result.get("suggestions", []),
            "answer": result.get("answer", ""),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
