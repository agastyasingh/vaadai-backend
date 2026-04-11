# Gunicorn — https://docs.gunicorn.org/en/stable/settings.html
#
# Latency vs concurrency:
# - Increasing `workers` runs more processes. That helps when MANY users hit the app at
#   once; it does NOT make a single /ask request faster (one request still uses one worker).
# - `threads` lets one worker handle several concurrent requests (I/O-bound). Helpful on
#   small hosts if users overlap; still does not shrink time for one isolated request.
# - Per-request speed is dominated by Claude + Indian Kanoon API calls (see parallel fetch
#   in claude_rag_test.py), not Gunicorn settings.
#
# Memory: each worker is a full Python process — too many workers on a 512MB instance
# can cause OOM. Start with 1–2 workers on free/small tiers.

timeout = 130

# Render/small VPS: 2 workers × 4 threads (requires gthread worker for thread pool).
workers = 2
threads = 4
worker_class = "gthread"
