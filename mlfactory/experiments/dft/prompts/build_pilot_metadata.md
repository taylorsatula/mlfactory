# Pilot metadata generation prompt

You write prompts and metadata for text-generation models. Given a piece of writing, produce:
1. A concise user prompt that could plausibly produce the writing.
2. The use_case: one of [blog post, news article, educational article, website copy, personal essay, product description].
3. The style: one of [conversational, clear and informative, professional, essayistic, warm, concise and modern].
Return ONLY a JSON object with keys: prompt, use_case, style.

{extra_instructions}
