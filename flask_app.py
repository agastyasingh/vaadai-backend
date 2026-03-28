import sys
import os

# Make sure our pipeline file is importable
sys.path.insert(0, '/home/agastyasingh927/mysite')

from flask import Flask, request, jsonify
from flask_cors import CORS
from claude_rag_test import rag_query

app = Flask(__name__)
CORS(app)  # allows your Hostinger frontend to call this API

# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "VaadAI is running"})

# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask():
    data     = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = rag_query(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
