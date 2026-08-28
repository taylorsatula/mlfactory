# Human escalation via Hermes (Discord)

`mlfactory ask-human` lets an autonomous overseer — a pi agent minding a long
training run, or an mlfactory plugin — text a human on Discord when it hits a
condition that genuinely needs a human decision, then **block until the human
replies**. The reply text comes back to the overseer so it can act on it. This
is the "bug that needs human intervention" path for unattended runs.

It rides on the Hermes Agent gateway (the Discord bot). Hermes already knows
how to send and receive Discord messages; this command adds the *wait for the
reply* half — and it releases the instant you reply, not on a poll cadence.

## One-time setup

### 1. Configure Discord and run the gateway

```bash
hermes gateway setup      # configure the discord bot token + home channel
hermes gateway install    # run the messaging gateway as a user service
hermes gateway start      # (or: systemctl --user start hermes-gateway)
hermes gateway status     # expect: ✓ User gateway service is running
```

The gateway must stay running while you want replies to be captured — it owns
the persistent Discord connection. `hermes gateway install` + systemd linger
makes it survive logout/reboot. Confirm Discord is connected and find your
escalation chat id:

```bash
hermes send --list --json      # lists every configured platform + chat ids
```

Use a DM with the bot (a Discord self-DM) as the escalation chat — a distinct
chat that never collides with your normal traffic, so an escalation reply is
unambiguous.

### 2. Make the escalation chat intake-only (REQUIRED)

**This is the step that prevents the Hermes bot from replying to your reply
(and, with tools, acting on it).** By default the gateway runs an agent turn
on every inbound message. For an escalation chat you want the opposite:
persist the human's reply so the poll can read it, but run **no** agent turn.

Set `HERMES_INTAKE_ONLY_CHATS` in `~/.hermes/.env` to a comma/space-separated
list of chat ids to make intake-only, then restart the gateway:

```bash
# ~/.hermes/.env
HERMES_INTAKE_ONLY_CHATS=1542606970899398686      # your discord DM channel id
# (add as many as you like; also readable from config.yaml: gateway.intake_only_chats)
```
```bash
systemctl --user restart hermes-gateway.service
```

Intake-only is per-chat: only the chats you list skip the agent turn. Every
other chat behaves exactly as before. Matching is exact string equality on the
chat id — the platform chat id is the same token used to address the chat.

### 3. Set the escalation target

The target is the chat escalations go to, as a `hermes send` target:

```bash
export MLFACTORY_ESCALATION_TARGET='discord:1542606970899398686'   # discord DM
```

Put this in the environment that launches the overseer (your shell rc, the
`nohup`/systemd unit that runs the training watcher, etc.) so the model can
call `mlfactory ask-human` without passing `--to` each time. You can always
override per-call with `--to`.

## Usage

### From a shell / overseer agent

```bash
mlfactory ask-human "LoRA fine-tune OOM'd at step 5000 with loss=nan. \
Restart with --grad_accum 8, or abort and dump the checkpoint?" \
    --timeout 86400
```

- Prints the human's reply on **stdout**; diagnostics on stderr. Exit `0` on
  reply, `124` on timeout, `1` on a send/config error.
- `--timeout` defaults to **24h**. Use `0` to wait forever. As a hard kill
  switch you can also wrap the call: `timeout 86400 mlfactory ask-human ...`
  (exits `124` the same way).
- `--json` emits `{"status":"ok"|"timeout"|"error", ...}` on stdout for
  programmatic callers.
- Read a long message from stdin with `-`: `echo "$body" | mlfactory ask-human -`.

### From an mlfactory plugin (Python)

```python
from mlfactory.core.human_escalation import escalate_to_human, EscalationTimeout

try:
    decision = escalate_to_human(
        "Checkpoint save failed: disk full at /mnt/runs. Free space and "
        "resume, or abort?",
        timeout=6 * 3600,          # block up to 6h
        # target=...,               # optional; defaults to $MLFACTORY_ESCALATION_TARGET
    )
except EscalationTimeout:
    decision = "abort"             # fall back when no human answers in time
# ...act on `decision`
```

`escalate_to_human()` raises `EscalationConfigError` (no target / gateway not
set up), `EscalationSendError` (delivery failed), or `EscalationTimeout`.

## How reply capture works (and what it does *not* do)

1. **Send.** The message is delivered with `hermes send --to <target>` — the
   same fire-and-forget path cron/CI use. It reuses the gateway's credentials
   and writes nothing to the session DB.
2. **Watermark.** After the send lands we record the max `messages.id` among
   user rows in `~/.hermes/state.db`, so any reply is strictly newer.
3. **Push (instant release).** The Hermes intake-only gate, after persisting
   your reply, writes a wake record to a per-chat FIFO at
   `~/.hermes/intake_only/<chat_id>.fifo`. `ask-human` is blocked reading that
   FIFO, so it wakes the instant the reply lands — no poll latency, even if
   you reply hours later. The write is non-blocking; if no overseer is
   listening it's dropped harmlessly.
4. **Poll (fallback).** A state.db poll runs as a backstop for the cases the
   FIFO can miss: the overseer started after the reply, the gateway restarted
   and the gate ran before the FIFO was opened, or the wake was dropped. The
   first one to find the reply wins.
5. **Match + return.** A reply matches if its session's `chat_id` equals the
   escalation chat id (exact equality). The gateway's own acknowledgement (an
   `assistant` row) is ignored — only the human's reply is returned.

We read the Hermes DB read-only (`?mode=ro`) and talk to Hermes only through
the stable `hermes send` CLI and the intake-only FIFO, so this is decoupled
from Hermes internals — all DB access lives in one module
(`mlfactory/core/human_escalation.py`).

**Why intake-only is required (not optional).** Without it the gateway runs
an agent turn on your reply: the bot messages back, and because that agent has
shell/file tools it could misinterpret your decision reply ("restart with
--grad_accum 8", "free space and resume") as an instruction and act on it.
Intake-only makes the escalation chat write-only-from-Hermes / read-only-to-
mlfactory: your reply is persisted for the poll, and the bot stays silent.

## Guarding an escalation (recommended)

Phrase the message so a short reply is a usable decision, and always keep a
fallback the overseer can take when the human never answers:

```bash
timeout 86400 mlfactory ask-human "..." || decision="best-effort: abort+checkpoint"
```

Record the escalation and its reply as a lab note in the run
(`mlfactory note <run_id> ...`) so the human-in-the-loop decision is part of
the run's provenance.
