"""Tests for mlfactory.core.human_escalation.

Covers target parsing, exact chat-id matching, and the send -> watermark ->
push/poll -> reply flow against a temporary Hermes-style state.db, with the
network (hermes send), the FIFO wait, and the clock/sleep injected so the
test is fast and hermetic.
"""
from __future__ import annotations

import itertools
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from mlfactory.core.human_escalation import (
    EscalationConfigError,
    EscalationSendError,
    EscalationTimeout,
    SendResult,
    _chat_id_matches,
    _find_reply,
    _max_user_message_id,
    canonical_target_key,
    escalate_to_human,
)


# ---------------------------------------------------------------------------
# Target parsing / matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plat,canonical,db_chat_id,expected", [
    # Discord/Slack/Telegram: bare snowflake, exact match.
    ("discord", "1542606970899398686", "1542606970899398686", True),
    ("discord", "1542606970899398686", "1542606970899398687", False),
    ("discord", "1542606970899398686", "", False),
    ("slack", "C0123ABCD", "C0123ABCD", True),
    ("telegram", "-1001234567890", "-1001234567890", True),
])
def test_chat_id_matches_exact(plat, canonical, db_chat_id, expected):
    assert _chat_id_matches(plat, canonical, db_chat_id) is expected


def test_chat_id_matches_none_db():
    assert _chat_id_matches("discord", "1542606970899398686", None) is False


def test_canonical_target_key():
    assert canonical_target_key("discord:1542606970899398686") == (
        "discord", "1542606970899398686",
    )
    assert canonical_target_key("slack:C0123ABCD") == ("slack", "C0123ABCD")
    with pytest.raises(EscalationConfigError):
        canonical_target_key("")
    with pytest.raises(EscalationConfigError):
        canonical_target_key("discord")  # no chat id


# ---------------------------------------------------------------------------
# Fixtures: a tiny Hermes-style state.db
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    chat_id TEXT,
    chat_type TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    timestamp REAL NOT NULL
);
"""


def _make_state_db(tmp: Path, seed: list[tuple[str, str, str, str, float]]) -> Path:
    """Create a state.db with seeded (session_id, chat_id, role, content, ts)."""
    path = tmp / "state.db"
    con = sqlite3.connect(path)
    con.executescript(_SCHEMA)
    sessions = {}
    for sid, chat_id, role, content, ts in seed:
        if sid not in sessions:
            con.execute("INSERT INTO sessions(id, chat_id, chat_type) VALUES(?,?,?)",
                        (sid, chat_id, "dm"))
            sessions[sid] = chat_id
        con.execute("INSERT INTO messages(session_id, role, content, timestamp) VALUES(?,?,?,?)",
                    (sid, role, content, ts))
    con.commit()
    con.close()
    return path


# A no-op FIFO wait so tests never block on a real pipe.
def _noop_wait(chat_id, deadline, clock, sleep):
    return None


# ---------------------------------------------------------------------------
# escalate_to_human: success, ignore-noise, timeout
# ---------------------------------------------------------------------------

def test_escalate_returns_first_matching_user_reply(tmp_path):
    target = "discord:1542606970899398686"
    chat = "1542606970899398686"
    # Pre-seed: an old reply in the escalation chat (id=1) and an unrelated
    # chat's user message (id=2). Watermark after send = max user id = 1.
    db = _make_state_db(tmp_path, [
        ("s_self", chat, "user", "previous escalation", 10.0),
        ("s_other", "1999999999999999999", "user", "someone else", 11.0),
    ])

    from mlfactory.core.human_escalation import _connect_ro

    calls = {"connect": 0, "send": []}

    def send_fn(t, msg, subj):
        calls["send"].append((t, msg, subj))
        return SendResult(True, "", {"success": True, "chat_id": chat})

    def connect_fn():
        calls["connect"] += 1
        # The human replies between the 2nd and 3rd poll.
        if calls["connect"] == 3:
            rw = sqlite3.connect(db)
            rw.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES(?,?,?,?)",
                       ("s_self", "user", "restart it", 12.0))
            # Gateway's own acknowledgement must be ignored (assistant row).
            rw.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES(?,?,?,?)",
                       ("s_self", "assistant", "got it, restarting", 13.0))
            # An unrelated chat's user message arriving in the same window.
            rw.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES(?,?,?,?)",
                       ("s_other", "user", "unrelated reply", 14.0))
            rw.commit()
            rw.close()
        return _connect_ro(db)

    clock = itertools.count(start=0, step=1)
    reply = escalate_to_human(
        "training crashed at step 5000; reply with how to proceed",
        target=target,
        timeout=1000,
        poll_interval=0,
        state_db=db,
        send_fn=send_fn,
        connect_fn=connect_fn,
        sleep_fn=lambda s: None,
        clock_fn=lambda: next(clock),
        wait_fn=_noop_wait,
    )

    assert reply == "restart it"
    assert calls["send"] and calls["send"][0][0] == target
    assert calls["send"][0][2] == "[mlfactory escalation]"


def test_escalate_times_out(tmp_path):
    chat = "1542606970899398686"
    db = _make_state_db(tmp_path, [("s_self", chat, "user", "old", 1.0)])

    from mlfactory.core.human_escalation import _connect_ro

    clock = itertools.count(start=0, step=1)  # ~5 polls before deadline passes
    with pytest.raises(EscalationTimeout):
        escalate_to_human(
            "waiting for a reply that never comes",
            target=f"discord:{chat}",
            timeout=5,
            poll_interval=0,
            state_db=db,
            send_fn=lambda t, m, s: SendResult(True, "", {"success": True, "chat_id": chat}),
            connect_fn=lambda: _connect_ro(db),
            sleep_fn=lambda s: None,
            clock_fn=lambda: next(clock),
            wait_fn=_noop_wait,
        )


def test_escalate_requires_target(monkeypatch):
    monkeypatch.delenv("MLFACTORY_ESCALATION_TARGET", raising=False)
    with pytest.raises(EscalationConfigError):
        escalate_to_human("hi", send_fn=lambda *a: None,
                          connect_fn=lambda: None, sleep_fn=lambda s: None,
                          clock_fn=lambda: 0.0, wait_fn=_noop_wait)


def test_escalate_send_failure_surfaces(tmp_path):
    def boom(t, m, s):
        raise EscalationSendError("delivery failed")
    with pytest.raises(EscalationSendError):
        escalate_to_human("hi", target="discord:1542606970899398686", send_fn=boom,
                          connect_fn=lambda: None, sleep_fn=lambda s: None,
                          clock_fn=lambda: 0.0, wait_fn=_noop_wait)


def test_send_raises_on_error_payload(monkeypatch):
    """An {"error": ...} payload from `hermes send` must fail fast, not
    silently turn into a poll-until-timeout."""
    from mlfactory.core import human_escalation as he

    class _P:
        returncode = 0
        stdout = '{"error": "platform not configured"}'
        stderr = ""

    monkeypatch.setattr(he.subprocess, "run", lambda *a, **k: _P())
    with pytest.raises(EscalationSendError) as ei:
        he._send("discord:1542606970899398686", "hi", None)
    assert "not configured" in str(ei.value)


# ---------------------------------------------------------------------------
# Push path: FIFO wait wakes the poll
# ---------------------------------------------------------------------------

def test_fifo_wait_drives_release(tmp_path):
    """The push wait_fn is called each loop; when it 'wakes' and the reply is
    in the DB, escalate returns it — proving the FIFO wait gates each poll,
    not the sleep."""
    chat = "1542606970899398686"
    db = _make_state_db(tmp_path, [("s_self", chat, "user", "old", 1.0)])

    from mlfactory.core.human_escalation import _connect_ro

    wait_calls = {"n": 0}

    def wait_fn(cid, deadline, clock, sleep):
        wait_calls["n"] += 1
        # On the 2nd wait, simulate the gate writing the reply to the DB.
        if wait_calls["n"] == 2:
            rw = sqlite3.connect(db)
            rw.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES(?,?,?,?)",
                       ("s_self", "user", "do it", 2.0))
            rw.commit()
            rw.close()

    reply = escalate_to_human(
        "escalation",
        target=f"discord:{chat}",
        timeout=1000,
        poll_interval=0,
        state_db=db,
        send_fn=lambda t, m, s: SendResult(True, "", {"success": True, "chat_id": chat}),
        connect_fn=lambda: _connect_ro(db),
        sleep_fn=lambda s: None,
        clock_fn=lambda: 0.0,
        wait_fn=wait_fn,
    )
    assert reply == "do it"
    assert wait_calls["n"] >= 2  # the wait was actually invoked


# ---------------------------------------------------------------------------
# Direct DB helpers
# ---------------------------------------------------------------------------

def test_max_user_message_id_and_find_reply(tmp_path):
    chat = "1542606970899398686"
    db = _make_state_db(tmp_path, [
        ("s_self", chat, "user", "old", 1.0),
        ("s_self", chat, "assistant", "ack", 2.0),
        ("s_other", "1999999999999999999", "user", "other", 3.0),
    ])
    from mlfactory.core.human_escalation import _connect_ro

    con = _connect_ro(db)
    assert _max_user_message_id(con) == 3  # max user row id is the 'other' one
    # The only user row on the escalation chat after id=0 is "old" (id=1).
    assert _find_reply(con, "discord", (chat,), 0) == (1, "old")
    # After the watermark, nothing on the escalation chat yet.
    assert _find_reply(con, "discord", (chat,), 1) is None
    con.close()


# ---------------------------------------------------------------------------
# Discord end-to-end (snowflake chat_id, exact match, send-result key)
# ---------------------------------------------------------------------------

def test_escalate_discord_ignores_other_chat(tmp_path):
    chat = "1542606970899398686"
    other = "9999999999999999999"
    db = _make_state_db(tmp_path, [("s_dm", chat, "user", "old", 1.0)])

    from mlfactory.core.human_escalation import _connect_ro

    def send_fn(t, m, s):
        return SendResult(True, "", {"success": True, "chat_id": chat})

    def connect_fn():
        # A reply lands on an UNRELATED discord DM — must not be returned.
        rw = sqlite3.connect(db)
        rw.execute(
            "INSERT INTO messages(session_id,role,content,timestamp) VALUES(?,?,?,?)",
            ("s_other", "user", "wrong chat", 2.0),
        )
        rw.execute("INSERT OR IGNORE INTO sessions(id,chat_id,chat_type) VALUES(?,?,?)",
                   ("s_other", other, "dm"))
        rw.commit()
        rw.close()
        return _connect_ro(db)

    clock = itertools.count(start=0, step=1)
    with pytest.raises(EscalationTimeout):
        escalate_to_human(
            "discord escalation",
            target=f"discord:{chat}",
            timeout=5,
            poll_interval=0,
            state_db=db,
            send_fn=send_fn,
            connect_fn=connect_fn,
            sleep_fn=lambda s: None,
            clock_fn=lambda: next(clock),
            wait_fn=_noop_wait,
        )


# ---------------------------------------------------------------------------
# CLI: mlfactory ask-human
# ---------------------------------------------------------------------------

def _cli(target_response, monkeypatch):
    """Build a CliRunner with `escalate_to_human` replaced by a stub."""
    import mlfactory.core.human_escalation as he

    calls = {}

    def fake(message, *, target=None, timeout=None, poll_interval=None,
             state_db=None, subject=None, **_):
        calls["kwargs"] = dict(message=message, target=target, timeout=timeout,
                               subject=subject)
        if isinstance(target_response, Exception):
            raise target_response
        return target_response

    monkeypatch.setattr(he, "escalate_to_human", fake)
    return CliRunner(), calls


def test_cli_success(monkeypatch):
    runner, calls = _cli("restart it", monkeypatch)
    from mlfactory.cli import main
    res = runner.invoke(main, ["ask-human", "hi there", "--to", "discord:1542606970899398686"])
    assert res.exit_code == 0
    assert res.output.strip() == "restart it"
    assert calls["kwargs"]["target"] == "discord:1542606970899398686"
    assert calls["kwargs"]["subject"] == "[mlfactory escalation]"


def test_cli_timeout_exit_124(monkeypatch):
    runner, _ = _cli(EscalationTimeout(30), monkeypatch)
    from mlfactory.cli import main
    res = runner.invoke(main, ["ask-human", "hi", "--to", "discord:1542606970899398686", "--timeout", "30"])
    assert res.exit_code == 124
    assert "timed out" in res.output.lower() or "timed out" in (res.stderr or "").lower()


def test_cli_error_exit_1(monkeypatch):
    runner, _ = _cli(EscalationConfigError("no target"), monkeypatch)
    from mlfactory.cli import main
    res = runner.invoke(main, ["ask-human", "hi", "--to", "discord:1542606970899398686"])
    assert res.exit_code == 1


def test_cli_timeout_zero_means_forever(monkeypatch):
    runner, calls = _cli("ok", monkeypatch)
    from mlfactory.cli import main
    res = runner.invoke(main, ["ask-human", "hi", "--to", "discord:1542606970899398686", "--timeout", "0"])
    assert res.exit_code == 0
    import math
    assert math.isinf(calls["kwargs"]["timeout"])


def test_cli_json_output(monkeypatch):
    runner, _ = _cli("the reply", monkeypatch)
    from mlfactory.cli import main
    res = runner.invoke(main, ["ask-human", "hi", "--to", "discord:1542606970899398686", "--json"])
    assert res.exit_code == 0
    import json as _json
    assert _json.loads(res.output) == {"status": "ok", "reply": "the reply"}
