"""Human-in-the-loop escalation via the Hermes Agent gateway.

This module lets an autonomous overseer (a pi agent minding a long training
run, or an mlfactory plugin) text a human on a messaging platform (Discord,
Slack, Telegram, …) when it hits a condition that genuinely needs a human
decision, then block until the human replies. The reply text is returned to
the caller so the overseer can act on it.

How it works
------------
1. **Send.** The message is delivered with ``hermes send --to <target>``,
   the same fire-and-forget path cron jobs and CI use. ``hermes send`` reuses
   the gateway's already-configured credentials and needs no LLM, no agent
   loop, and no running gateway to *send*.
2. **Wait.** A reply can only arrive while the Hermes gateway is running,
   because it owns the persistent connection to the platform (the Discord
   gateway socket, etc.). The gateway writes every inbound message into its
   session store (``~/.hermes/state.db``): one row in ``messages``
   (``role='user'``, ``content``, autoincrement ``id``) belonging to a
   ``sessions`` row keyed by the platform chat id (a bare snowflake such as
   a Discord channel id).
3. **Push + poll.** After sending we record a watermark (the max
   ``messages.id`` of any user row) and block on a per-chat notification FIFO
   that the Hermes intake-only gate writes to the instant it persists a reply
   — so release is immediate, with no poll latency. A state.db poll runs as a
   fallback (covers a missed FIFO notification: the overseer started after the
   reply, the gateway restarted, etc.). The first one to find the reply wins.
4. **Match.** A reply matches if its session's ``chat_id`` equals the
   escalation chat id (exact string equality — the platform chat id is the
   same token used to address the chat). The gateway's own acknowledgement
   (an ``assistant`` row) is ignored — only the human's reply is returned.
5. **Intake-only.** The escalation chat must be configured intake-only on
   the Hermes side (``HERMES_INTAKE_ONLY_CHATS``) so the gateway persists the
   human's reply but does NOT run an agent turn on it — otherwise the bot
   would reply to the reply (and, with tools, could act on it). See
   ``docs/HUMAN_ESCALATION.md``.

This is deliberately decoupled from Hermes internals: we read the session DB
read-only (``?mode=ro``), talk to Hermes only through the stable ``hermes
send`` CLI and the intake-only notification FIFO, and confine all DB access to
this module so a future schema change only touches one function.

One-time setup is documented in ``docs/HUMAN_ESCALATION.md``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "EscalationError",
    "EscalationConfigError",
    "EscalationSendError",
    "EscalationTimeout",
    "canonical_target_key",
    "escalate_to_human",
]


class EscalationError(Exception):
    """Base class for escalation failures."""


class EscalationConfigError(EscalationError):
    """The escalation target or Hermes state DB is not configured."""


class EscalationSendError(EscalationError):
    """``hermes send`` failed to deliver the message."""


class EscalationTimeout(EscalationError):
    """No reply arrived within the timeout."""

    def __init__(self, timeout: float):
        super().__init__(f"no reply within {timeout:g}s")
        self.timeout = timeout


# ---------------------------------------------------------------------------
# Target normalization / matching
# ---------------------------------------------------------------------------
#
# Escalation targets are ``hermes send`` strings: ``platform:chat_id``.
# Reply matching compares the canonical chat id against the ``chat_id`` of
# inbound ``role='user'`` rows in the Hermes session DB. Discord, Slack, and
# Telegram all use bare snowflake chat ids (e.g. ``1542606970899398686``),
# matched by exact string equality — the ``chat_id`` the bridge reports is
# the same token used to address the chat, so no normalization is needed.


def _parse_target(target: str) -> tuple[str, str]:
    """Split a ``platform:chat_id`` target into (platform, chat_id_ref)."""
    t = target.strip()
    if ":" in t:
        plat, _, ref = t.partition(":")
        return plat.strip().lower(), ref.strip()
    return t.lower(), ""


def canonical_target_key(target: str) -> tuple[str, str]:
    """Return ``(platform, canonical_chat_id)`` for a ``hermes send`` target.

    The chat id is the bare ref after ``platform:``, stripped of whitespace.
    The pair is the match key for replies.
    """
    if not target:
        raise EscalationConfigError("empty escalation target")
    plat, ref = _parse_target(target)
    if not ref:
        raise EscalationConfigError(f"escalation target has no chat id: {target!r}")
    return plat, ref.strip()


def _chat_id_matches(platform: str, canonical_key: str, db_chat_id: str) -> bool:
    """True if a state.db ``sessions.chat_id`` is the escalation chat.

    Exact string equality — the platform chat id is the same token used to
    address the chat, so the DB row's ``chat_id`` equals the target's chat id
    for a matching reply.
    """
    if not db_chat_id:
        return False
    return str(db_chat_id).strip() == str(canonical_key).strip()


# ---------------------------------------------------------------------------
# Hermes state DB access (read-only)
# ---------------------------------------------------------------------------

def _default_state_db_path() -> Path:
    explicit = os.environ.get("HERMES_STATE_DB")
    if explicit:
        return Path(explicit).expanduser()
    hermes_dir = os.environ.get("HERMES_DIR") or os.environ.get("HERMES_HOME")
    if hermes_dir:
        return Path(hermes_dir).expanduser() / "state.db"
    return Path.home() / ".hermes" / "state.db"


def _connect_ro(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise EscalationConfigError(
            f"Hermes state DB not found at {path}. Is the Hermes gateway "
            "running and the escalation platform connected? "
            "See docs/HUMAN_ESCALATION.md."
        )
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5.0)
    con.row_factory = sqlite3.Row
    return con


def _max_user_message_id(con: sqlite3.Connection) -> int:
    """High-water mark: the largest ``messages.id`` among user rows, or 0."""
    row = con.execute(
        "SELECT COALESCE(MAX(id), 0) AS m FROM messages WHERE role = 'user'"
    ).fetchone()
    return int(row["m"]) if row else 0


# ---------------------------------------------------------------------------
# Push notification FIFO (zero-lag release)
# ---------------------------------------------------------------------------
#
# The Hermes intake-only gate writes a one-line JSON record to a per-chat FIFO
# at ``<hermes_home>/intake_only/<chat_id>.fifo`` the instant it persists a
# reply. We block on reading it so release is immediate (no poll latency). The
# state.db poll runs as a fallback for the cases the FIFO can miss: the
# overseer started after the reply, the gateway restarted and the gate ran
# before we opened the FIFO, or the write was dropped (no reader at the time).

def _intake_fifo_path(chat_id: str) -> Path:
    hermes_dir = os.environ.get("HERMES_DIR") or os.environ.get("HERMES_HOME")
    base = Path(hermes_dir).expanduser() if hermes_dir else Path.home() / ".hermes"
    return base / "intake_only" / f"{chat_id}.fifo"


def _wait_for_fifo(
    chat_id: str,
    deadline: float,
    clock: callable,
    sleep: callable,
) -> None:
    """Block on the intake-only FIFO until notified or the deadline nears.

    Returns on (a) a wake record written by the gate, or (b) the deadline.
    Never raises — a FIFO failure just falls through to the poll path. Re-arms
    on each wake so multiple replies in one wait still work.

    Opens the FIFO non-blocking for read and ``select``s for readability: a FIFO
    with no writer is *not* readable, so ``select`` blocks (up to the remaining
    deadline) until a writer opens the other end and writes — that is the
    instant-wake signal. A writer that opens then closes (EOF) re-arms the next
    iteration.
    """
    path = _intake_fifo_path(chat_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            try:
                os.mkfifo(path)
            except FileExistsError:
                pass
    except Exception:
        return
    import select

    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            return
        fd = None
        try:
            # O_RDONLY | O_NONBLOCK: open returns immediately even with no
            # writer; the fd is only *readable* once a writer opens and writes.
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            r, _, _ = select.select([fd], [], [], min(remaining, 30.0))
            if not r:
                # Timed out this slice with no writer — fall through to the
                # poll, which will re-enter this wait on the next loop tick.
                return
            data = os.read(fd, 65536)
            if data:
                return  # wake — a record was written
            # EOF (writer opened then closed without writing, or closed after):
            # re-arm for the next writer.
        except FileNotFoundError:
            return
        except Exception:
            return
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass


def _find_reply(
    con: sqlite3.Connection,
    platform: str,
    match_keys: tuple[str, ...],
    after_id: int,
) -> Optional[tuple[int, str]]:
    """First new user message on the escalation chat, as (id, content).

    ``match_keys`` are the canonical chat IDs the reply may arrive under (the
    ``chat_id`` the send resolved to, plus the target-derived key). A reply
    matches if its session's ``chat_id`` equals any of them (exact string
    equality — the platform chat id is the same token used to address the
    chat).
    """
    rows = con.execute(
        """
        SELECT m.id AS mid, m.content AS content, s.chat_id AS chat_id
        FROM messages m
        JOIN sessions s ON m.session_id = s.id
        WHERE m.role = 'user' AND m.id > ?
        ORDER BY m.id ASC
        """,
        (after_id,),
    ).fetchall()
    for r in rows:
        chat_id = r["chat_id"] or ""
        if any(_chat_id_matches(platform, k, chat_id) for k in match_keys):
            content = r["content"]
            if content is None:
                continue
            return int(r["mid"]), str(content)
    return None


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

@dataclass
class SendResult:
    success: bool
    raw: str
    payload: dict


def _send(target: str, message: str, subject: Optional[str]) -> SendResult:
    """Deliver via ``hermes send`` and report success/failure."""
    cmd = ["hermes", "send", "--to", target, "--json"]
    if subject:
        cmd += ["--subject", subject]
    try:
        proc = subprocess.run(
            cmd,
            input=message,
            text=True,
            capture_output=True,
            timeout=120,
        )
    except FileNotFoundError:
        raise EscalationConfigError(
            "the `hermes` CLI was not found on PATH. Install Hermes Agent and "
            "ensure `hermes` is executable."
        )
    except subprocess.TimeoutExpired:
        raise EscalationSendError("`hermes send` did not return within 120s")

    raw = proc.stdout.strip()
    payload: dict = {}
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = {}

    # Error shape from send_message_tool is {"error": "..."}; surface it
    # immediately rather than letting the caller poll until timeout.
    if "error" in payload:
        raise EscalationSendError(f"`hermes send` failed: {payload['error']}")
    # Success shape is {"success": true, ...}; fall back to the exit code for
    # any non-JSON-but-zero-exit delivery.
    success = bool(payload.get("success")) or proc.returncode == 0
    if not success:
        hint = raw or proc.stderr.strip() or f"exit {proc.returncode}"
        raise EscalationSendError(f"`hermes send` failed: {hint}")
    return SendResult(True, raw, payload)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def escalate_to_human(
    message: str,
    *,
    target: Optional[str] = None,
    timeout: float = 86400.0,
    poll_interval: float = 5.0,
    state_db: Optional[os.PathLike | str] = None,
    subject: Optional[str] = "[mlfactory escalation]",
    send_fn: Optional[callable] = None,
    connect_fn: Optional[callable] = None,
    sleep_fn: Optional[callable] = None,
    clock_fn: Optional[callable] = None,
    wait_fn: Optional[callable] = None,
) -> str:
    """Send an escalation via Hermes and block until the human replies.

    Parameters
    ----------
    message:
        The text shown to the human. Phrase it so a short reply is a usable
        decision (the first reply on the chat is what comes back).
    target:
        ``hermes send`` target for the escalation chat. Defaults to the
        ``MLFACTORY_ESCALATION_TARGET`` env var. Use ``discord:<channel-id>``
        for a Discord DM/channel (recommended), or the equivalent for
        Slack/Telegram.
    timeout:
        Maximum seconds to wait for a reply. Default 24 h. The caller can also
        wrap the call in ``timeout <s>`` as a hard kill switch; the command
        exits 124 either way.
    poll_interval:
        Seconds between state.db fallback polls. Default 5.
    state_db:
        Override path to the Hermes session DB. Defaults to
        ``$HERMES_STATE_DB`` / ``$HERMES_DIR/state.db`` / ``~/.hermes/state.db``.
    subject:
        Optional subject line prepended to the message (default
        ``"[mlfactory escalation]"``). Pass ``None`` for no prefix.

    Returns
    -------
    str
        The human's reply text (the first new ``role='user'`` message on the
        escalation chat after the send).

    Raises
    ------
    EscalationConfigError, EscalationSendError, EscalationTimeout
    """
    if not message or not message.strip():
        raise EscalationConfigError("no escalation message provided")

    target = target or os.environ.get("MLFACTORY_ESCALATION_TARGET")
    if not target:
        raise EscalationConfigError(
            "no escalation target. Set --to / MLFACTORY_ESCALATION_TARGET to "
            "e.g. discord:<channel-id>. See docs/HUMAN_ESCALATION.md."
        )

    platform, canonical = canonical_target_key(target)
    db_path = Path(state_db).expanduser() if state_db else _default_state_db_path()

    send = send_fn or _send
    result = send(target, message, subject)

    # The send result carries the authoritative chat_id the bridge actually
    # used. Add it to the match set so a reply is matched even if it differs
    # in form from the target (it shouldn't for snowflake ids, but be safe).
    match_keys = {canonical}
    sent_chat_id = None
    if isinstance(result, SendResult):
        sent_chat_id = result.payload.get("chat_id")
    if sent_chat_id:
        try:
            _plat, sent_key = canonical_target_key(f"{platform}:{sent_chat_id}")
            match_keys.add(sent_key)
        except EscalationConfigError:
            pass
    match_keys_t = tuple(match_keys)
    # The push notification FIFO is keyed on the chat id the gateway sees.
    fifo_chat_id = sent_chat_id or canonical

    connect = connect_fn or (lambda p=db_path: _connect_ro(p))
    sleep = sleep_fn or time.sleep
    clock = clock_fn or time.monotonic
    wait = wait_fn or _wait_for_fifo

    # Watermark *after* the send lands so any reply is strictly newer.
    con = connect()
    try:
        watermark = _max_user_message_id(con)
    finally:
        con.close()

    deadline = clock() + timeout
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise EscalationTimeout(timeout)
        # Push path: block on the intake-only FIFO for a near-instant wake,
        # capped at the remaining deadline. Falls through to the poll on wake
        # (or timeout) — the poll is what actually reads + returns the reply.
        wait(fifo_chat_id, deadline, clock, sleep)
        con = connect()
        try:
            found = _find_reply(con, platform, match_keys_t, watermark)
        finally:
            con.close()
        if found is not None:
            _id, content = found
            return content
        sleep(min(poll_interval, max(remaining, 0.0)))
