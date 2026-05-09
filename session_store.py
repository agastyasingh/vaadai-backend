"""
Cross-worker session store backed by SQLite.

Why SQLite (not in-memory dicts):
- Gunicorn runs N worker processes; in-memory state is per-process and silently
  diverges across requests. SQLite gives one shared, atomic, file-locked store.

What a "session" tracks:
- session_id        : uuid for the current burst of activity
- history           : conversation turns (user/assistant) for this session
- suggestion_map    : reply_id -> {text, ...} for the latest follow-up chips sent
- question_count    : how many user questions have been ASKED in this session (cap = 8)
- last_active_at    : monotonic-ish wall clock; > 10 min idle => new session
- processing        : per-user lock so background workers don't race

Sessions roll over (new session_id, cleared history/suggestions/count) when:
- the user sends a message AND last_active_at is older than SESSION_TIMEOUT_SECONDS, OR
- they hit the 8-question cap (next message starts a new session).
"""

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.environ.get("VAADAI_SESSION_DB", "/tmp/vaadai_sessions.db")
SESSION_TIMEOUT_SECONDS = 10 * 60
MAX_QUESTIONS_PER_SESSION = 8
QUEUE_STALENESS_SECONDS = 10 * 60  # discard queued items older than this

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    user_phone        TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    created_at        REAL NOT NULL,
    last_active_at    REAL NOT NULL,
    question_count    INTEGER NOT NULL DEFAULT 0,
    suggestions_json  TEXT NOT NULL DEFAULT '{}',
    history_json      TEXT NOT NULL DEFAULT '[]',
    processing        INTEGER NOT NULL DEFAULT 0,
    processing_since  REAL
);

CREATE TABLE IF NOT EXISTS message_queue (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_phone    TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    text          TEXT NOT NULL,
    is_suggested  INTEGER NOT NULL DEFAULT 0,
    enqueued_at   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_queue_user ON message_queue(user_phone, id);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


@contextmanager
def _txn():
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE;")
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.close()


def _new_session_row(user_phone: str, now: float) -> dict:
    return {
        "user_phone": user_phone,
        "session_id": str(uuid.uuid4()),
        "created_at": now,
        "last_active_at": now,
        "question_count": 0,
        "suggestions_json": "{}",
        "history_json": "[]",
        "processing": 0,
        "processing_since": None,
    }


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["history"] = json.loads(d.pop("history_json") or "[]")
    d["suggestions"] = json.loads(d.pop("suggestions_json") or "{}")
    return d


def _upsert(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO sessions
            (user_phone, session_id, created_at, last_active_at,
             question_count, suggestions_json, history_json,
             processing, processing_since)
        VALUES (:user_phone, :session_id, :created_at, :last_active_at,
                :question_count, :suggestions_json, :history_json,
                :processing, :processing_since)
        ON CONFLICT(user_phone) DO UPDATE SET
            session_id        = excluded.session_id,
            created_at        = excluded.created_at,
            last_active_at    = excluded.last_active_at,
            question_count    = excluded.question_count,
            suggestions_json  = excluded.suggestions_json,
            history_json      = excluded.history_json,
            processing        = excluded.processing,
            processing_since  = excluded.processing_since
        """,
        row,
    )


def get_or_rotate_session(user_phone: str, now: Optional[float] = None) -> dict:
    """
    Return the active session for this user, rotating to a fresh one if the
    previous session has been idle longer than SESSION_TIMEOUT_SECONDS or if
    it has hit MAX_QUESTIONS_PER_SESSION.

    Updates last_active_at to `now`.
    """
    now = now if now is not None else time.time()

    with _txn() as conn:
        cur = conn.execute(
            "SELECT * FROM sessions WHERE user_phone = ?", (user_phone,)
        )
        row = cur.fetchone()

        if row is None:
            fresh = _new_session_row(user_phone, now)
            _upsert(conn, fresh)
            return _row_to_dict(_fetch(conn, user_phone))

        existing = _row_to_dict(row)
        idle = now - existing["last_active_at"]
        capped = existing["question_count"] >= MAX_QUESTIONS_PER_SESSION

        if idle > SESSION_TIMEOUT_SECONDS or capped:
            # Drop any stale queued messages from the old session.
            conn.execute(
                "DELETE FROM message_queue WHERE user_phone = ?", (user_phone,)
            )
            fresh = _new_session_row(user_phone, now)
            _upsert(conn, fresh)
            return _row_to_dict(_fetch(conn, user_phone))

        existing["last_active_at"] = now
        _save(conn, existing)
        return _row_to_dict(_fetch(conn, user_phone))


def _fetch(conn: sqlite3.Connection, user_phone: str) -> sqlite3.Row:
    return conn.execute(
        "SELECT * FROM sessions WHERE user_phone = ?", (user_phone,)
    ).fetchone()


def _save(conn: sqlite3.Connection, sess: dict) -> None:
    conn.execute(
        """
        UPDATE sessions SET
            session_id        = :session_id,
            created_at        = :created_at,
            last_active_at    = :last_active_at,
            question_count    = :question_count,
            suggestions_json  = :suggestions_json,
            history_json      = :history_json,
            processing        = :processing,
            processing_since  = :processing_since
        WHERE user_phone = :user_phone
        """,
        {
            "user_phone": sess["user_phone"],
            "session_id": sess["session_id"],
            "created_at": sess["created_at"],
            "last_active_at": sess["last_active_at"],
            "question_count": sess["question_count"],
            "suggestions_json": json.dumps(sess["suggestions"]),
            "history_json": json.dumps(sess["history"]),
            "processing": sess["processing"],
            "processing_since": sess["processing_since"],
        },
    )


def lookup_suggestion(user_phone: str, reply_id: str) -> Optional[str]:
    """Return the suggestion text the user tapped, or None if reply_id is unknown/stale."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT suggestions_json FROM sessions WHERE user_phone = ?",
            (user_phone,),
        ).fetchone()
    if not row:
        return None
    sugs = json.loads(row["suggestions_json"] or "{}")
    entry = sugs.get(reply_id)
    if isinstance(entry, dict):
        return entry.get("text")
    if isinstance(entry, str):
        return entry
    return None


def store_suggestions(user_phone: str, suggestions: list) -> dict:
    """Replace the per-user suggestion map. Returns the new map ({id: text})."""
    mapping = {f"suggestion_{i}": s for i, s in enumerate(suggestions[:10])}
    with _txn() as conn:
        row = _fetch(conn, user_phone)
        if row is None:
            return mapping
        sess = _row_to_dict(row)
        sess["suggestions"] = {k: {"text": v} for k, v in mapping.items()}
        _save(conn, sess)
    return mapping


def enqueue_message(
    user_phone: str, session_id: str, text: str, is_suggested: bool
) -> int:
    """Append a message to this user's FIFO queue. Returns the queue row id."""
    now = time.time()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO message_queue
                (user_phone, session_id, text, is_suggested, enqueued_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_phone, session_id, text, 1 if is_suggested else 0, now),
        )
        return cur.lastrowid


def try_acquire_processing_lock(user_phone: str) -> bool:
    """
    Atomically claim the per-user processing slot. Returns True if we got it,
    False if another worker already holds it.

    A stale lock (held >2 minutes) is force-released — covers worker crashes.
    """
    now = time.time()
    with _txn() as conn:
        row = _fetch(conn, user_phone)
        if row is None:
            return False
        sess = _row_to_dict(row)
        if sess["processing"] and sess["processing_since"] and (
            now - sess["processing_since"] < 120
        ):
            return False
        sess["processing"] = 1
        sess["processing_since"] = now
        _save(conn, sess)
        return True


def release_processing_lock(user_phone: str) -> None:
    with _txn() as conn:
        row = _fetch(conn, user_phone)
        if row is None:
            return
        sess = _row_to_dict(row)
        sess["processing"] = 0
        sess["processing_since"] = None
        _save(conn, sess)


def release_lock_if_queue_empty(user_phone: str) -> bool:
    """
    Atomically check the queue under transaction; release the processing lock
    only if no items remain. This closes the race where a new message arrives
    between the worker's last pop_next_message() and its lock release.

    Returns True if released (queue truly empty), False if work remains —
    in which case the caller should loop and pop again.
    """
    cutoff = time.time() - QUEUE_STALENESS_SECONDS
    with _txn() as conn:
        conn.execute(
            "DELETE FROM message_queue WHERE user_phone = ? AND enqueued_at < ?",
            (user_phone, cutoff),
        )
        pending = conn.execute(
            "SELECT 1 FROM message_queue WHERE user_phone = ? LIMIT 1",
            (user_phone,),
        ).fetchone()
        if pending is not None:
            return False
        row = _fetch(conn, user_phone)
        if row is None:
            return True
        sess = _row_to_dict(row)
        sess["processing"] = 0
        sess["processing_since"] = None
        _save(conn, sess)
        return True


def pop_next_message(user_phone: str) -> Optional[dict]:
    """
    Pop the oldest queued message for this user that is still fresh. Stale
    items (older than QUEUE_STALENESS_SECONDS) are discarded silently — these
    are typically WhatsApp webhook redeliveries from hours ago.

    Returns {id, session_id, text, is_suggested, enqueued_at} or None.
    """
    cutoff = time.time() - QUEUE_STALENESS_SECONDS
    with _txn() as conn:
        # Discard stale items first.
        conn.execute(
            "DELETE FROM message_queue WHERE user_phone = ? AND enqueued_at < ?",
            (user_phone, cutoff),
        )
        row = conn.execute(
            """
            SELECT id, session_id, text, is_suggested, enqueued_at
            FROM message_queue
            WHERE user_phone = ?
            ORDER BY id ASC
            LIMIT 1
            """,
            (user_phone,),
        ).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM message_queue WHERE id = ?", (row["id"],))
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "text": row["text"],
            "is_suggested": bool(row["is_suggested"]),
            "enqueued_at": row["enqueued_at"],
        }


def increment_question_count(user_phone: str) -> int:
    """Increment and return the new question count for the user's current session."""
    with _txn() as conn:
        row = _fetch(conn, user_phone)
        if row is None:
            return 0
        sess = _row_to_dict(row)
        sess["question_count"] += 1
        sess["last_active_at"] = time.time()
        _save(conn, sess)
        return sess["question_count"]


def append_history(user_phone: str, user_text: str, assistant_text: str) -> list:
    with _txn() as conn:
        row = _fetch(conn, user_phone)
        if row is None:
            return []
        sess = _row_to_dict(row)
        sess["history"].append({"role": "user", "content": user_text})
        sess["history"].append({"role": "assistant", "content": assistant_text})
        sess["last_active_at"] = time.time()
        _save(conn, sess)
        return sess["history"]


def get_history(user_phone: str) -> list:
    with _connect() as conn:
        row = conn.execute(
            "SELECT history_json FROM sessions WHERE user_phone = ?",
            (user_phone,),
        ).fetchone()
    if row is None:
        return []
    return json.loads(row["history_json"] or "[]")


def get_session_snapshot(user_phone: str) -> Optional[dict]:
    with _connect() as conn:
        row = _fetch(conn, user_phone)
    return _row_to_dict(row) if row else None


# Initialize the schema at import time so workers don't race on first request.
init_db()
