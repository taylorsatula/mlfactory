# Voice Fine-Tune — Context

## The plan

Fine-tune **Qwen3.5-4B** so that how Taylor writes to customers is internalized in
the model weights rather than carried by system-prompt steering. The deployment
target is MIRA's SMS autonomy loop: the fine-tuned model becomes the generator for
autonomous customer replies.

Today, voice is carried entirely by prompt-side steering — distilled "business
voice directives" injected into the system prompt (the confidence-gated autonomy
loop described in `docs/BUSINESS_VOICE_LEARNING.md` in the crm_mira repo). After
the fine-tune, the model's default behavior should already be Taylor's voice. The
directives stay, but as dressing and confirmation on top: edge-case guidance
landing on a model that already sounds right, rather than the mechanism that
constructs the voice from scratch every generation.

## Why weights, not (only) prompts

- Prompt steering captures **explicit** patterns — "lead with empathy on
  complaints," "use 'I' not 'we'," "don't over-explain pricing." Most of a voice is
  **tacit**: rhythm, brevity, greeting habits, how prices are framed, how delays
  are owned, when a thumbs-up is the whole reply. Thousands of implicit patterns
  never distill into prose directives.
- Every steering token is context-window and attention tax paid on every
  generation, forever.
- A model that defaults to Taylor's voice degrades gracefully — uncovered
  situations fall back toward his style instead of toward generic corporate-speak.
- The capture → distill → inject directive loop continues to run after the
  fine-tune; its job shifts from building the voice to refining it.

## The data

`data/threads/` holds **449 JSON files, one per customer thread** — the entire
usable SMS/iMessage history between Taylor's personal phone and phone numbers
present in his Square customer list.

**Provenance.** Extracted 2026-08-04 from an unencrypted iPhone backup
(`sms.db`, modern iOS 17/18-era schema) on Taylor's Mac, matched against a Square
customer CSV export (646 rows; 609 had usable 10-digit numbers). The extractor is
`scripts/extract_imessage_corpus.py` in the crm_mira repo (Mac-side; it reads the
backup directly). This corpus predates MIRA — it spans effectively the whole
history of the business, 2024-04-12 → 2026-08-04.

**Cleaning already applied (do not re-do):**

- 1:1 threads only — 264 group chats excluded (voice attribution ambiguity).
- Tapbacks/reactions removed (`associated_message_type != 0`).
- 1,393 null-text messages recovered by decoding `attributedBody` typedstream
  blobs, including the iOS edited-message NSMutableString layout; 0 decode
  failures in the final corpus.
- Multiple iOS chats sharing one phone number merged into a single thread
  (`chat_count` records how many).
- Chronological order, Apple-epoch timestamps converted to UTC ISO-8601.
- Attachment-only messages dropped — this corpus is text only, no MMS media.

**Scale and shape.** 6,173 messages total: 3,406 from Taylor, 2,767 from
customers. Thread sizes are heavily skewed — 138 threads have fewer than 3
messages; the largest has 475. The median message is short (p50 ≈ 65 chars),
which is faithful to the channel: much of Taylor's voice lives in brief
logistical replies, not just long explanations.

**Schema (per file):**

```json
{
  "customer": {"name": "...", "reference_id": "...", "square_customer_id": "...", "phone_raw": "+1..."},
  "normalized_number": "2565551234",
  "chat_count": 2,
  "message_count": 71,
  "first_message_utc": "2024-09-16T14:52:54Z",
  "last_message_utc": "2026-05-11T14:26:07Z",
  "messages": [{"ts": "2024-09-16T14:52:54Z", "from": "customer", "text": "..."}]
}
```

`from` is `"taylor"` (the voice being learned — outgoing, `is_from_me=1`) or
`"customer"` (incoming context). Names/numbers come from the Square export, not
from the phone's contacts.

## Caveats and open questions

- **PII.** Real customer names and phone numbers throughout. `data/` is covered
  by mlfactory's `.gitignore`; keep it out of git, off shared instances, and out
  of any artifact that leaves this machine.
- **Two flagged threads, decision pending** (both matched the Square export, so
  both are included): `2318385007_taylor-satula.json` (161 msgs — appears to be
  Taylor's own/family number sitting in the customer list) and
  `4704298635_josiah-quality-care-exteriors.json` (475 msgs — the largest thread,
  but reads like a vendor contact rather than a customer).
- **Authenticity.** Nothing was rewritten. Two messages consist of a lone U+FFFD
  — that is how iOS stored them, kept as-is. Taylor's side includes typos,
  voice-to-text artifacts, and mid-conversation register shifts; that is the
  signal, not noise.
- **Channel blend.** Threads mix iMessage and SMS over the years; the export does
  not distinguish them per message.

## Local Qwen3-1.7B proof of concept

A local supervised LoRA POC is wired through the `voice-train` stage:

- Spec: `specs/voice_qwen3_1_7b_poc_v2.yaml`
- Trainer/plugin: `train_voice.py` and `train_plugin.py`
- Base model: local `/home/admin/models/hf/Qwen3-1.7B`
- Raw threads remain local; examples are redacted in memory and raw SMS files are not copied into run artifacts.
- The POC uses thread-level train/test separation and generic before/after prompts. It is an engineering proof that the adapter can alter response style, not a production-quality voice or privacy release gate.

The review and training stages remain separate: the POC does not treat placeholder text as a training target and drops examples whose target contains a redaction marker.
