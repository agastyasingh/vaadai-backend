import sys
import os

# Make sure our pipeline file is importable
sys.path.insert(0, '/home/agastyasingh927/mysite')

from flask import Flask, request, jsonify
from flask_cors import CORS
from vaad_ai_ik_rag import rag_query

app = Flask(__name__)
CORS(app)  # allows your Hostinger frontend to call this API

# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "VaadAI is running"})

# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.route("/debug-cases", methods=["GET"])
def debug_cases():
    """Temporary debug endpoint to test IK web scraping."""
    import requests, re, html as html_lib
    from urllib.parse import quote_plus
    query_str  = "eviction non-payment rent Maharashtra sortby:mostrecent"
    search_url = "https://indiankanoon.org/search/?formInput=" + quote_plus(query_str)
    try:
        resp = requests.get(
            search_url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            timeout=15,
        )
        titles  = re.findall(r'<a\s+href="/docfragment/(\d+)/[^"]*"[^>]*>\s*([^<]+?)\s*</a>', resp.text)
        courts  = re.findall(r'<span\s+class="docsource">\s*([^<]+?)\s*</span>', resp.text)
        cases   = [{"title": html_lib.unescape(t), "source": courts[i] if i < len(courts) else ""} 
                   for i, (_, t) in enumerate(titles[:3])]
        return jsonify({"url": search_url, "status": resp.status_code, "cases": cases, "html_snippet": resp.text[5000:6000]})
    except Exception as e:
        return jsonify({"error": str(e), "url": search_url})


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
