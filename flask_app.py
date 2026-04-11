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

@app.route("/ask", methods=["POST"])
def ask():
    data     = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    history  = data.get("history", [])

    if not question:
        return jsonify({"error": "No question provided"}), 400

    clean_history = [
        {"role": t["role"], "content": t["content"]}
        for t in history
        if isinstance(t, dict)
        and t.get("role") in ("user", "assistant")
        and isinstance(t.get("content"), str)
    ]

    try:
        result = rag_query(question, history=clean_history)
        # result is now {"answer": str, "suggestions": list}
        return jsonify({
            "answer":      result.get("answer", ""),
            "suggestions": result.get("suggestions", [])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
