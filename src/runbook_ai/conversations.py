"""Tiny SQLite store for local conversation history."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    """Persist local chat turns so a page refresh does not erase the conversation."""

    def __init__(self, path="data/conversations.sqlite3"):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self):
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_conversation_idx
                    ON messages(conversation_id, id);
                """
            )

    def create(self, title="New conversation") -> str:
        conversation_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title[:80], now, now),
            )
        return conversation_id

    def list_conversations(self, limit=20) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_messages(self, conversation_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT role, content, metadata_json, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                **json.loads(row["metadata_json"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def add_message(self, conversation_id: str, role: str, content: str, **metadata) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Only user and assistant messages can be stored.")
        now = _now()
        with self._connect() as db:
            exists = db.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if exists is None:
                raise KeyError("Conversation does not exist.")
            db.execute(
                "INSERT INTO messages(conversation_id, role, content, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, json.dumps(metadata, ensure_ascii=False), now),
            )
            if role == "user" and exists["title"] == "New conversation":
                title = " ".join(content.split())[:72]
                db.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (title or "New conversation", now, conversation_id),
                )
            else:
                db.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )

    def clear(self, conversation_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            db.execute(
                "UPDATE conversations SET title = 'New conversation', updated_at = ? WHERE id = ?",
                (_now(), conversation_id),
            )

    def delete(self, conversation_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

