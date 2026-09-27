"""UZA 엔진: 선제 대화, 사용자 대화, 하루 정리, 주간 검토, 기억 통제.

docs/IMPLEMENTATION.md 2장 흐름을 그대로 따른다.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

from . import context as ctxb
from . import gate
from .config import Config
from .llm import LLM, LLMError
from .prompts import MESSAGE_TYPES
from .store import Store

log = logging.getLogger(__name__)

CONTINUED_TURNS = 3
SHORT_REPLY_CHARS = 10
LONG_REPLY_CHARS = 80
CURRENT_STATE_DEFAULT_DAYS = 3
CURRENT_STATE_MAX_DAYS = 14
DND_MAX_DAYS = 30
LEVELS = ["mentioned", "repeated", "confirmed"]
CHAT_FALLBACK = "앗, 잠깐 말이 꼬였어. 한 번만 다시 말해줄래?"
HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class Messenger(Protocol):
    def send(self, text: str) -> None: ...


@dataclass
class ProactiveOutcome:
    decision: str
    reason: str
    message_type: str | None = None
    message: str | None = None


def _response_length(text: str) -> str:
    n = len(text.strip())
    if n <= SHORT_REPLY_CHARS:
        return "short"
    if n >= LONG_REPLY_CHARS:
        return "long"
    return "normal"


def _matches_forgotten(text: str, forgotten: list[str]) -> bool:
    t = text.strip()
    return any(f and (f in t or t in f) for f in (x.strip() for x in forgotten))


class Uza:
    def __init__(self, cfg: Config, store: Store, llm: LLM, messenger: Messenger):
        self.cfg = cfg
        self.store = store
        self.llm = llm
        self.messenger = messenger

    # ── 2.1 선제 대화 ─────────────────────────────────────
    def proactive_tick(self, now: datetime) -> ProactiveOutcome:
        # 9장: 다음 선제 판단 시점까지 답이 없으면 무응답 확정
        for pending in self.store.pending_sent_logs():
            self.store.update_proactive_log(pending.id, replied=False)

        g = gate.check(self.cfg, self.store, now)
        if not g.passed:
            return self._skip(now, g.reason)

        context = ctxb.build(self.cfg, self.store, now, "proactive")
        try:
            out = self.llm.complete("proactive", context)
        except LLMError as e:
            log.warning("proactive 호출 실패: %s", e)
            return self._skip(now, "llm_error")

        reason = str(out.get("reason") or "")
        if out.get("decision") != "SEND":
            return self._skip(now, reason or "model_skip")

        message = str(out.get("message") or "").strip()
        mtype = out.get("message_type")
        if not message or mtype not in MESSAGE_TYPES:
            return self._skip(now, "invalid_output")

        # 같은 유형이 연속으로 나가지 않게 코드에서도 막는다.
        last_sent = self.store.recent_sent_logs(1)
        if last_sent and last_sent[0].message_type == mtype:
            return self._skip(now, "repeat_type")

        self.messenger.send(message)
        mid = self.store.add_message(now, "uza", message, is_proactive=True)
        self.store.add_proactive_log(now, "SEND", reason, mtype, mid)
        return ProactiveOutcome("SEND", reason, mtype, message)

    def _skip(self, now: datetime, reason: str) -> ProactiveOutcome:
        self.store.add_proactive_log(now, "SKIP", reason)
        return ProactiveOutcome("SKIP", reason)

    # ── 2.2 사용자 대화 ───────────────────────────────────
    def handle_user_message(self, text: str, now: datetime) -> str:
        self.store.add_message(now, "user", text)
        self._track_reaction(text, now)

        context = ctxb.build(self.cfg, self.store, now, "chat")
        context["user_message"] = text
        try:
            out = self.llm.complete("chat", context)
        except LLMError as e:
            log.warning("chat 호출 실패: %s", e)
            out = {"message": CHAT_FALLBACK}

        reply = str(out.get("message") or "").strip() or CHAT_FALLBACK

        cmd = out.get("memory_command")
        if isinstance(cmd, dict):
            if cmd.get("action") == "forget":
                self.forget(str(cmd.get("target") or ""), list(cmd.get("target_ids") or []), now)
            elif cmd.get("action") == "list":
                reply = f"{reply}\n\n{self.describe_memory(now)}"

        if dnd := out.get("do_not_disturb_until"):
            self._set_dnd(str(dnd), now)
        if (freq := out.get("preferred_frequency")) in ("low", "normal", "high"):
            self.store.update_settings(preferred_frequency=freq)

        self.messenger.send(reply)
        self.store.add_message(now, "uza", reply)
        self._track_continuation()
        return reply

    def _track_reaction(self, text: str, now: datetime) -> None:
        for pending in self.store.pending_sent_logs():
            delay = int((now - pending.timestamp).total_seconds() // 60)
            self.store.update_proactive_log(
                pending.id,
                replied=True,
                reply_delay_minutes=delay,
                response_length=_response_length(text),
            )

    def _track_continuation(self) -> None:
        last = self.store.recent_sent_logs(1)
        if not last or not last[0].replied or last[0].conversation_continued or last[0].message_id is None:
            return
        if len(self.store.messages_after(last[0].message_id)) >= CONTINUED_TURNS:
            self.store.update_proactive_log(last[0].id, conversation_continued=True)

    def _set_dnd(self, value: str, now: datetime) -> None:
        try:
            until = datetime.fromisoformat(value)
        except ValueError:
            log.warning("do_not_disturb_until 파싱 실패: %r", value)
            return
        if until.tzinfo is None:
            until = until.replace(tzinfo=self.cfg.timezone)
        until = until.astimezone(self.cfg.timezone)
        if until <= now:
            return
        until = min(until, now + timedelta(days=DND_MAX_DAYS))
        self.store.update_settings(do_not_disturb_until=until)

    # ── 8. 기억 통제 ─────────────────────────────────────
    def forget(self, target: str, target_ids: list[str], now: datetime) -> list[str]:
        """해당 episode / hypothesis / current_state를 지운다. 지운 문구 목록을 돌려준다."""
        removed: list[str] = []
        for tid in target_ids:
            kind, _, num = str(tid).partition(":")
            if not num.isdigit():
                continue
            if kind == "episode" and (e := self.store.get_episode(int(num))):
                self.store.delete_episode(e.id)
                removed.append(e.summary)
            elif kind == "hypothesis" and (h := self.store.get_hypothesis(int(num))):
                self.store.delete_hypothesis(h.id)
                removed.append(h.statement)

        target = target.strip()
        if not removed and target:
            for e in self.store.active_episodes(now):
                if target in e.summary:
                    self.store.delete_episode(e.id)
                    removed.append(e.summary)
            for h in self.store.hypotheses():
                if target in h.statement:
                    self.store.delete_hypothesis(h.id)
                    removed.append(h.statement)
        if target:
            for s in self.store.active_current_state(now):
                if target in s.statement:
                    self.store.delete_current_state(s.id)
                    removed.append(s.statement)

        for text in dict.fromkeys(removed + ([target] if target else [])):
            self.store.add_forgotten(text, now)
        return removed

    def describe_memory(self, now: datetime) -> str:
        """repeated 이상 가설과 미해결 episode를 평범한 말로. 숨기지 않는다."""
        lines = []
        for h in self.store.hypotheses():
            if h.level in ("repeated", "confirmed"):
                lines.append(f"- {h.statement}")
        for e in self.store.active_episodes(now):
            if not e.resolved:
                when = f" ({e.related_date:%m/%d})" if e.related_date else ""
                lines.append(f"- {e.summary}{when}")
        for s in self.store.active_current_state(now):
            lines.append(f"- 요즘: {s.statement}")
        if not lines:
            return "아직 따로 기억해 둔 건 없어."
        return "내가 기억하고 있는 건 이 정도야:\n" + "\n".join(lines) + "\n지우고 싶은 게 있으면 말해줘."

    # ── 2.3 하루 정리 ─────────────────────────────────────
    def reflect(self, now: datetime) -> dict | None:
        """직전 하루(now - 24h ~ now) 대화를 정리한다. 대화가 없으면 건너뛴다."""
        day = now.date()
        messages = self.store.messages_between(now - timedelta(days=1), now + timedelta(seconds=1))
        if not messages:
            return None

        open_eps = [e for e in self.store.active_episodes(now) if not e.resolved]
        hyps = self.store.hypotheses()
        forgotten = self.store.forgotten()
        context = {
            "date": day.isoformat(),
            "today_messages": [
                {"at": f"{m.timestamp:%H:%M}", "by": m.sender, "text": m.text} for m in messages
            ],
            "open_episodes": [ctxb._episode_dict(e) for e in open_eps],
            "existing_hypotheses": [
                {"id": h.id, "statement": h.statement, "level": h.level, "sensitive": h.sensitive}
                for h in hyps
            ],
            "current_state": [s.statement for s in self.store.active_current_state(now)],
            "forgotten": forgotten,
        }
        try:
            out = self.llm.complete("reflection", context)
        except LLMError as e:
            log.warning("reflection 호출 실패: %s", e)
            return None

        self._apply_reflection(out, day, now, {e.id for e in open_eps}, forgotten)
        out["date"] = day.isoformat()
        self.store.save_reflection(day, out)
        return out

    def _apply_reflection(
        self, out: dict, day: date, now: datetime, open_ids: set[int], forgotten: list[str]
    ) -> None:
        for ev in out.get("important_events") or []:
            summary = str(ev.get("summary") or "").strip()
            if not summary or _matches_forgotten(summary, forgotten):
                continue
            related = None
            if ev.get("related_date"):
                try:
                    related = date.fromisoformat(str(ev["related_date"])[:10])
                except ValueError:
                    related = None
            importance = ev.get("importance") if ev.get("importance") in ("low", "mid", "high") else "mid"
            resolved = related is None or related < day
            if related:
                expires = datetime.combine(related + timedelta(days=7), datetime.min.time(), self.cfg.timezone)
            else:
                days = 60 if importance == "high" else 30
                expires = now + timedelta(days=days)
            self.store.add_episode(day, summary, related, resolved, importance, expires)

        for eid in out.get("resolved_episode_ids") or []:
            if eid in open_ids:
                self.store.resolve_episode(eid)

        for cs in out.get("current_state") or []:
            statement = str(cs.get("statement") or "").strip()
            if not statement or _matches_forgotten(statement, forgotten):
                continue
            days = cs.get("expires_in_days")
            days = CURRENT_STATE_DEFAULT_DAYS if not isinstance(days, int) or days < 1 else days
            days = min(days, CURRENT_STATE_MAX_DAYS)
            self.store.add_current_state(statement, now, now + timedelta(days=days))

        for upd in out.get("hypothesis_updates") or []:
            self._apply_hypothesis_update(upd, day, forgotten)

    def _apply_hypothesis_update(self, upd: dict, day: date, forgotten: list[str]) -> None:
        statement = str(upd.get("statement") or "").strip()
        observation = upd.get("observation")
        stamp = day.isoformat()

        h = self.store.get_hypothesis(upd["id"]) if isinstance(upd.get("id"), int) else None
        if h is None and statement:
            h = next((x for x in self.store.hypotheses() if x.statement == statement), None)
            if h is not None and observation == "new":
                observation = "support"

        if h is None:
            if observation != "new" or not statement or _matches_forgotten(statement, forgotten):
                return
            # 새 가설은 무조건 mentioned에서 시작한다. 한 번으로 단정하지 않는다.
            self.store.add_hypothesis(statement, [stamp], day, bool(upd.get("sensitive")))
            return

        if observation == "support":
            if stamp not in h.evidence:
                h.evidence.append(stamp)
            h.last_seen = day
            if upd.get("user_stated"):
                h.level = "confirmed"
            elif h.level == "mentioned" and len(set(h.evidence)) >= 2:
                h.level = "repeated"
        elif observation == "contradict":
            if stamp not in h.contradictions:
                h.contradictions.append(stamp)
            h.last_seen = day
        else:
            return
        if upd.get("sensitive"):
            h.sensitive = True
        self.store.save_hypothesis(h)

    # ── 2.4 주간 검토 ─────────────────────────────────────
    def review(self, now: datetime) -> dict | None:
        self.store.purge_expired_state(now)
        week_ago = now - timedelta(days=7)
        rate, samples = gate.response_rate_7d(self.store, now)
        start, end = gate.active_hours(self.cfg, self.store)
        hours = Counter(t.hour for t in self.store.user_message_times(week_ago))
        hyps = self.store.hypotheses()

        context = {
            "now": ctxb.fmt_dt(now),
            "reflections": self.store.reflections_since(week_ago.date()),
            "hypotheses": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "level": h.level,
                    "evidence_count": len(h.evidence),
                    "contradiction_count": len(h.contradictions),
                    "last_seen": h.last_seen.isoformat() if h.last_seen else None,
                    "sensitive": h.sensitive,
                }
                for h in hyps
            ],
            "proactive_stats": {
                "sent_7d": len(self.store.sent_logs_since(week_ago)),
                "response_rate_7d": rate,
                "samples": samples,
                "daily_max_now": gate.daily_max(self.cfg, self.store, now),
            },
            "active_hours": {"start": f"{start:%H:%M}", "end": f"{end:%H:%M}"},
            "user_message_hours": {f"{h:02d}": n for h, n in sorted(hours.items())},
        }
        try:
            out = self.llm.complete("review", context)
        except LLMError as e:
            log.warning("review 호출 실패: %s", e)
            return None

        touched: set[int] = set()
        for ch in out.get("hypothesis_changes") or []:
            h = self.store.get_hypothesis(ch.get("id")) if isinstance(ch.get("id"), int) else None
            if h is None:
                continue
            action = ch.get("action")
            touched.add(h.id)
            if action == "delete":
                self.store.delete_hypothesis(h.id)
                continue
            if action == "upgrade":
                h.level = LEVELS[min(LEVELS.index(h.level) + 1, len(LEVELS) - 1)]
            elif action == "downgrade":
                h.level = LEVELS[max(LEVELS.index(h.level) - 1, 0)]
            elif action == "rewrite" and ch.get("new_statement"):
                # 바뀐 문장은 새 주장이므로 처음부터 다시 쌓는다.
                h.statement = str(ch["new_statement"]).strip()
                h.level = "mentioned"
                h.evidence = [now.date().isoformat()]
                h.contradictions = []
            self.store.save_hypothesis(h)

        # 5.3: 반박이 근거보다 많은데 모델이 손대지 않은 가설은 코드가 지운다.
        for h in self.store.hypotheses():
            if h.id not in touched and len(h.contradictions) > len(h.evidence):
                self.store.delete_hypothesis(h.id)

        sugg = out.get("active_hours_suggestion")
        if isinstance(sugg, dict) and HHMM.match(str(sugg.get("start"))) and HHMM.match(str(sugg.get("end"))):
            self.store.update_settings(active_start=sugg["start"], active_end=sugg["end"])
        return out
