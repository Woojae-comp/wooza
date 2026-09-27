"""SQLite 저장소 (docs/IMPLEMENTATION.md 5장).

시각은 모두 설정 타임존 기준 ISO 문자열로 저장한다.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    sender TEXT NOT NULL CHECK (sender IN ('user', 'uza')),
    text TEXT NOT NULL,
    is_proactive INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    summary TEXT NOT NULL,
    related_date TEXT,
    resolved INTEGER NOT NULL DEFAULT 0,
    importance TEXT NOT NULL DEFAULT 'mid' CHECK (importance IN ('low', 'mid', 'high')),
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statement TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'mentioned'
        CHECK (level IN ('mentioned', 'repeated', 'confirmed')),
    evidence TEXT NOT NULL DEFAULT '[]',
    contradictions TEXT NOT NULL DEFAULT '[]',
    last_seen TEXT,
    sensitive INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS current_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statement TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proactive_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('SEND', 'SKIP')),
    reason TEXT,
    message_type TEXT,
    message_id INTEGER REFERENCES messages(id),
    replied INTEGER,                 -- NULL: 반응 대기, 1: 답함, 0: 무응답 확정
    reply_delay_minutes INTEGER,
    response_length TEXT,
    conversation_continued INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    active_start TEXT,
    active_end TEXT,
    do_not_disturb_until TEXT,
    preferred_frequency TEXT NOT NULL DEFAULT 'normal'
        CHECK (preferred_frequency IN ('low', 'normal', 'high'))
);

CREATE TABLE IF NOT EXISTS reflections (
    date TEXT PRIMARY KEY,
    data TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS forgotten (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

INSERT OR IGNORE INTO user_settings (id) VALUES (1);
"""


def iso(dt: datetime | date | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat(timespec="seconds") if isinstance(dt, datetime) else dt.isoformat()


def parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value[:10]) if value else None


@dataclass
class Message:
    id: int
    timestamp: datetime
    sender: str
    text: str
    is_proactive: bool


@dataclass
class Episode:
    id: int
    date: date
    summary: str
    related_date: date | None
    resolved: bool
    importance: str
    expires_at: datetime | None


@dataclass
class Hypothesis:
    id: int
    statement: str
    level: str
    evidence: list[str]
    contradictions: list[str]
    last_seen: date | None
    sensitive: bool


@dataclass
class CurrentState:
    id: int
    statement: str
    created_at: datetime
    expires_at: datetime


@dataclass
class ProactiveLog:
    id: int
    timestamp: datetime
    decision: str
    reason: str | None
    message_type: str | None
    message_id: int | None
    replied: bool | None
    reply_delay_minutes: int | None
    response_length: str | None
    conversation_continued: bool


@dataclass
class UserSettings:
    active_start: str | None
    active_end: str | None
    do_not_disturb_until: datetime | None
    preferred_frequency: str


def _bool(v: Any) -> bool | None:
    return None if v is None else bool(v)


class Store:
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ── 내부 ──────────────────────────────────────────────
    def _exec(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def _all(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    # ── messages ─────────────────────────────────────────
    def add_message(self, ts: datetime, sender: str, text: str, is_proactive: bool = False) -> int:
        cur = self._exec(
            "INSERT INTO messages (timestamp, sender, text, is_proactive) VALUES (?, ?, ?, ?)",
            (iso(ts), sender, text, int(is_proactive)),
        )
        return cur.lastrowid

    @staticmethod
    def _msg(r: sqlite3.Row) -> Message:
        return Message(r["id"], parse_dt(r["timestamp"]), r["sender"], r["text"], bool(r["is_proactive"]))

    def messages_between(self, start: datetime, end: datetime) -> list[Message]:
        rows = self._all(
            "SELECT * FROM messages WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp, id",
            (iso(start), iso(end)),
        )
        return [self._msg(r) for r in rows]

    def recent_messages(self, limit: int) -> list[Message]:
        rows = self._all("SELECT * FROM messages ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,))
        return [self._msg(r) for r in reversed(rows)]

    def last_message(self) -> Message | None:
        msgs = self.recent_messages(1)
        return msgs[0] if msgs else None

    def messages_after(self, message_id: int) -> list[Message]:
        rows = self._all("SELECT * FROM messages WHERE id > ? ORDER BY id", (message_id,))
        return [self._msg(r) for r in rows]

    def get_message(self, message_id: int) -> Message | None:
        rows = self._all("SELECT * FROM messages WHERE id = ?", (message_id,))
        return self._msg(rows[0]) if rows else None

    def user_message_times(self, since: datetime) -> list[datetime]:
        rows = self._all(
            "SELECT timestamp FROM messages WHERE sender = 'user' AND timestamp >= ?", (iso(since),)
        )
        return [parse_dt(r["timestamp"]) for r in rows]

    # ── episodes ─────────────────────────────────────────
    def add_episode(
        self,
        day: date,
        summary: str,
        related_date: date | None,
        resolved: bool,
        importance: str,
        expires_at: datetime | None,
    ) -> int:
        cur = self._exec(
            "INSERT INTO episodes (date, summary, related_date, resolved, importance, expires_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (iso(day), summary, iso(related_date), int(resolved), importance, iso(expires_at)),
        )
        return cur.lastrowid

    @staticmethod
    def _ep(r: sqlite3.Row) -> Episode:
        return Episode(
            r["id"],
            parse_date(r["date"]),
            r["summary"],
            parse_date(r["related_date"]),
            bool(r["resolved"]),
            r["importance"],
            parse_dt(r["expires_at"]),
        )

    def active_episodes(self, now: datetime) -> list[Episode]:
        rows = self._all(
            "SELECT * FROM episodes WHERE expires_at IS NULL OR expires_at > ?"
            " ORDER BY date DESC, id DESC",
            (iso(now),),
        )
        return [self._ep(r) for r in rows]

    def resolve_episode(self, episode_id: int) -> None:
        self._exec("UPDATE episodes SET resolved = 1 WHERE id = ?", (episode_id,))

    def get_episode(self, episode_id: int) -> Episode | None:
        rows = self._all("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        return self._ep(rows[0]) if rows else None

    def delete_episode(self, episode_id: int) -> None:
        self._exec("DELETE FROM episodes WHERE id = ?", (episode_id,))

    # ── hypotheses ───────────────────────────────────────
    def add_hypothesis(self, statement: str, evidence: list[str], last_seen: date, sensitive: bool) -> int:
        cur = self._exec(
            "INSERT INTO hypotheses (statement, level, evidence, last_seen, sensitive)"
            " VALUES (?, 'mentioned', ?, ?, ?)",
            (statement, json.dumps(evidence), iso(last_seen), int(sensitive)),
        )
        return cur.lastrowid

    @staticmethod
    def _hyp(r: sqlite3.Row) -> Hypothesis:
        return Hypothesis(
            r["id"],
            r["statement"],
            r["level"],
            json.loads(r["evidence"]),
            json.loads(r["contradictions"]),
            parse_date(r["last_seen"]),
            bool(r["sensitive"]),
        )

    def hypotheses(self) -> list[Hypothesis]:
        return [self._hyp(r) for r in self._all("SELECT * FROM hypotheses ORDER BY id")]

    def get_hypothesis(self, hid: int) -> Hypothesis | None:
        rows = self._all("SELECT * FROM hypotheses WHERE id = ?", (hid,))
        return self._hyp(rows[0]) if rows else None

    def save_hypothesis(self, h: Hypothesis) -> None:
        self._exec(
            "UPDATE hypotheses SET statement = ?, level = ?, evidence = ?, contradictions = ?,"
            " last_seen = ?, sensitive = ? WHERE id = ?",
            (
                h.statement,
                h.level,
                json.dumps(h.evidence),
                json.dumps(h.contradictions),
                iso(h.last_seen),
                int(h.sensitive),
                h.id,
            ),
        )

    def delete_hypothesis(self, hid: int) -> None:
        self._exec("DELETE FROM hypotheses WHERE id = ?", (hid,))

    # ── current_state ────────────────────────────────────
    def add_current_state(self, statement: str, created_at: datetime, expires_at: datetime) -> int:
        cur = self._exec(
            "INSERT INTO current_state (statement, created_at, expires_at) VALUES (?, ?, ?)",
            (statement, iso(created_at), iso(expires_at)),
        )
        return cur.lastrowid

    def active_current_state(self, now: datetime) -> list[CurrentState]:
        rows = self._all(
            "SELECT * FROM current_state WHERE expires_at > ? ORDER BY created_at DESC", (iso(now),)
        )
        return [
            CurrentState(r["id"], r["statement"], parse_dt(r["created_at"]), parse_dt(r["expires_at"]))
            for r in rows
        ]

    def purge_expired_state(self, now: datetime) -> int:
        return self._exec("DELETE FROM current_state WHERE expires_at <= ?", (iso(now),)).rowcount

    def delete_current_state(self, sid: int) -> None:
        self._exec("DELETE FROM current_state WHERE id = ?", (sid,))

    # ── proactive_log ────────────────────────────────────
    def add_proactive_log(
        self,
        ts: datetime,
        decision: str,
        reason: str | None,
        message_type: str | None = None,
        message_id: int | None = None,
    ) -> int:
        cur = self._exec(
            "INSERT INTO proactive_log (timestamp, decision, reason, message_type, message_id)"
            " VALUES (?, ?, ?, ?, ?)",
            (iso(ts), decision, reason, message_type, message_id),
        )
        return cur.lastrowid

    @staticmethod
    def _plog(r: sqlite3.Row) -> ProactiveLog:
        return ProactiveLog(
            r["id"],
            parse_dt(r["timestamp"]),
            r["decision"],
            r["reason"],
            r["message_type"],
            r["message_id"],
            _bool(r["replied"]),
            r["reply_delay_minutes"],
            r["response_length"],
            bool(r["conversation_continued"]),
        )

    def sent_logs_since(self, since: datetime) -> list[ProactiveLog]:
        rows = self._all(
            "SELECT * FROM proactive_log WHERE decision = 'SEND' AND timestamp >= ?"
            " ORDER BY timestamp, id",
            (iso(since),),
        )
        return [self._plog(r) for r in rows]

    def recent_sent_logs(self, limit: int) -> list[ProactiveLog]:
        rows = self._all(
            "SELECT * FROM proactive_log WHERE decision = 'SEND' ORDER BY timestamp DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [self._plog(r) for r in reversed(rows)]

    def all_logs(self) -> list[ProactiveLog]:
        return [self._plog(r) for r in self._all("SELECT * FROM proactive_log ORDER BY id")]

    def pending_sent_logs(self) -> list[ProactiveLog]:
        rows = self._all(
            "SELECT * FROM proactive_log WHERE decision = 'SEND' AND replied IS NULL ORDER BY id"
        )
        return [self._plog(r) for r in rows]

    def update_proactive_log(self, log_id: int, **fields: Any) -> None:
        allowed = {"replied", "reply_delay_minutes", "response_length", "conversation_continued"}
        if not fields or set(fields) - allowed:
            raise ValueError(f"수정할 수 없는 필드: {set(fields) - allowed}")
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = [int(v) if isinstance(v, bool) else v for v in fields.values()]
        self._exec(f"UPDATE proactive_log SET {cols} WHERE id = ?", (*values, log_id))

    # ── user_settings ────────────────────────────────────
    def settings(self) -> UserSettings:
        r = self._all("SELECT * FROM user_settings WHERE id = 1")[0]
        return UserSettings(
            r["active_start"], r["active_end"], parse_dt(r["do_not_disturb_until"]), r["preferred_frequency"]
        )

    def update_settings(self, **fields: Any) -> None:
        allowed = {"active_start", "active_end", "do_not_disturb_until", "preferred_frequency"}
        if not fields or set(fields) - allowed:
            raise ValueError(f"수정할 수 없는 필드: {set(fields) - allowed}")
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = [iso(v) if isinstance(v, datetime) else v for v in fields.values()]
        self._exec(f"UPDATE user_settings SET {cols} WHERE id = 1", values)

    # ── reflections ──────────────────────────────────────
    def save_reflection(self, day: date, data: dict) -> None:
        self._exec(
            "INSERT OR REPLACE INTO reflections (date, data) VALUES (?, ?)",
            (iso(day), json.dumps(data, ensure_ascii=False)),
        )

    def reflections_since(self, since: date) -> list[dict]:
        rows = self._all("SELECT data FROM reflections WHERE date >= ? ORDER BY date", (iso(since),))
        return [json.loads(r["data"]) for r in rows]

    def has_reflection(self, day: date) -> bool:
        return bool(self._all("SELECT 1 FROM reflections WHERE date = ?", (iso(day),)))

    # ── forgotten ────────────────────────────────────────
    def add_forgotten(self, text: str, ts: datetime) -> None:
        self._exec("INSERT INTO forgotten (text, created_at) VALUES (?, ?)", (text, iso(ts)))

    def forgotten(self) -> list[str]:
        return [r["text"] for r in self._all("SELECT text FROM forgotten ORDER BY id")]
