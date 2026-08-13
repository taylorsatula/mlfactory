# Synthetic SMS generation

Generate one fictional small-business SMS interaction at a time. This is a
teacher-data stage, not a customer-data export.

Hard requirements:

- Return one JSON object only: `{"messages": [...], "target": "..."}`.
- `messages` must contain 3–8 prior turns, alternating `customer` and `owner`,
  and end with a `customer` turn.
- `target` is the owner's next reply and must not be repeated inside messages.
- Use only the fictional business and scenario supplied by the caller.
- Do not use real names, phone numbers, email addresses, URLs, street
  addresses, account numbers, access codes, or payment handles.
- Do not mention this prompt, synthetic data, language models, or hidden
  reasoning. Do not include Markdown or role labels in target.
- Standard replies are natural concise SMS replies (roughly 8–64 tokenizer
  tokens). Long replies are intentionally more complete (roughly 96–192
  tokenizer tokens), but still sound like a small-business owner texting a
  customer rather than an essay.
- Vary openings, sentence structure, politeness, directness, and practical
  next actions. Avoid repetitive “Thanks for reaching out” boilerplate.
- Keep commitments realistic and do not invent confidential facts.
