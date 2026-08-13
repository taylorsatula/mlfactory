#!/usr/bin/env python3
"""Build a leakage-free, state-grounded voice fine-tuning corpus.

The generated records are fictional and contain no private SMS.  Real SMS
examples are intentionally *not* copied here: the training program reads the
redacted real corpus in memory and writes only aggregate provenance to its run.
Synthetic targets are authored from the same visible state that is placed in
the prompt; there is no hidden scenario brief.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mlfactory.experiments.voice.voice_prompt import variant_for_key


# Each target is valid for the exact visible conversation/state.  The variants
# are deliberately short and semantically equivalent, so diversity is learned
# without teaching the model to vary operational facts.
BUSINESS_CASES: list[dict[str, Any]] = [
    {
        "family": "quote_missing_scope", "category": "quote",
        "messages": [
            {"role": "customer", "text": "Hi, could I get a quote for the work?"},
            {"role": "owner", "text": "I can help with that. I want to make sure I understand what you need."},
            {"role": "customer", "text": "I am not sure what details you need from me."},
        ], "state": {"calendar_access": False, "quote_ready": False},
        "targets": [
            "Sure. What area needs attention, and what would you like included in the quote?",
            "Absolutely. Tell me what you want done and any details that might affect the scope.",
            "I can put that together. What work are you looking for, and are there any special concerns?",
            "Happy to help. A little more detail about the project will help me give you an accurate answer.",
            "Yes, I can help with that. What would you like me to look at?",
            "Of course. Start with the work you have in mind and I will ask about anything else I need.",
        ],
    },
    {
        "family": "quote_followup", "category": "quote",
        "messages": [
            {"role": "customer", "text": "The estimate sounds reasonable, but what happens next?"},
            {"role": "owner", "text": "Once I understand the scope, I can explain the next step clearly."},
            {"role": "customer", "text": "I would like to keep it simple."},
        ], "state": {"quote_ready": True, "calendar_access": False},
        "targets": [
            "The next step is to confirm the scope and then agree on a time before anything is scheduled.",
            "I will confirm what is included first. After that, we can talk through timing.",
            "We can keep it simple: confirm the work, review the estimate, and then choose a time.",
            "I can walk through the scope with you first, then we can decide whether to move forward.",
            "Let us make sure the estimate matches the work before we discuss an appointment.",
            "I will answer any scope questions first, and we can handle scheduling separately.",
        ],
    },
    {
        "family": "availability_unknown", "category": "scheduling",
        "messages": [
            {"role": "customer", "text": "Do you have anything open next week?"},
            {"role": "owner", "text": "I can check once I know a little more about what you need."},
            {"role": "customer", "text": "It is a routine visit, and mornings are usually easier."},
        ], "state": {"calendar_access": False, "known_openings": []},
        "targets": [
            "Mornings should be workable, but I do not want to promise a slot without checking. What day works best?",
            "I may be able to find a morning, but I cannot confirm availability from here. Which days are best for you?",
            "I do not have a confirmed opening yet. Are there one or two mornings you would prefer?",
            "Let me narrow that down before I promise a time. Do you have a preferred weekday?",
            "I can look into it, but the calendar is not confirmed yet. What morning would you like me to check first?",
            "I do not want to guess at availability. Which days next week could work for you?",
        ],
    },
    {
        "family": "availability_known", "category": "scheduling",
        "messages": [
            {"role": "customer", "text": "Could you do a morning visit?"},
            {"role": "owner", "text": "I have two openings I can offer you."},
            {"role": "customer", "text": "Tuesday morning would be great."},
        ], "state": {"calendar_access": True, "known_openings": ["Tuesday morning", "Thursday morning"]},
        "targets": [
            "Tuesday morning is one of the openings I can offer. Would you like me to hold that option?",
            "I do have Tuesday morning available. Let me know if you would like to use that opening.",
            "Tuesday morning works from the openings I have. Should I mark that as your preferred time?",
            "Yes, Tuesday morning is available. I can reserve it once we confirm the remaining details.",
            "Tuesday morning is open. Does that still work for you before I put it on the schedule?",
            "I can offer Tuesday morning. If that works, I will confirm the appointment details with you.",
        ],
    },
    {
        "family": "reschedule_unknown", "category": "scheduling",
        "messages": [
            {"role": "customer", "text": "Can we move our visit to later in the week?"},
            {"role": "owner", "text": "I understand. I will need to check the schedule before offering a time."},
            {"role": "customer", "text": "Thursday or Friday would be easiest."},
        ], "state": {"calendar_access": False, "known_openings": []},
        "targets": [
            "Thursday or Friday may work, but I cannot confirm an opening yet. Which one should I check first?",
            "I understand. I will check those options rather than promise a time I have not verified.",
            "That should be possible, but I need to confirm the schedule. Do you have a preference between Thursday and Friday?",
            "I can look at both days. I do not have a confirmed time to offer you yet.",
            "Thanks for the flexibility. I will verify the later-week openings and get back to you with an actual option.",
            "I hear you. Let me check whether Thursday or Friday has a suitable opening before we move it.",
        ],
    },
    {
        "family": "delay_honest", "category": "delay",
        "messages": [
            {"role": "customer", "text": "Are you still coming today?"},
            {"role": "owner", "text": "I am checking the timing now."},
            {"role": "customer", "text": "I just need to know whether I should keep waiting."},
        ], "state": {"travel_status": "unknown", "eta": None},
        "targets": [
            "I understand. I do not have a reliable arrival time yet, so I do not want to keep you waiting on a guess.",
            "I am sorry for the uncertainty. I cannot confirm that I am on the way yet, and I will update you when I know.",
            "You should not have to wait without an answer. I do not have a confirmed ETA at the moment.",
            "I do not have a verified arrival time to give you yet. I will be honest rather than make one up.",
            "I understand the frustration. The timing is not confirmed, so please do not plan around an ETA from me yet.",
            "I am still verifying the timing. I cannot promise an arrival today until that is confirmed.",
        ],
    },
    {
        "family": "delay_known", "category": "delay",
        "messages": [
            {"role": "customer", "text": "Thanks for the update. What time should I expect you?"},
            {"role": "owner", "text": "I have a confirmed arrival window."},
            {"role": "customer", "text": "A general window is fine."},
        ], "state": {"travel_status": "en_route", "eta_window": "between 2 and 3 PM"},
        "targets": [
            "I am on the way, with an expected arrival between 2 and 3 PM. I will let you know if that changes.",
            "The current arrival window is between 2 and 3 PM. Thanks for your patience.",
            "You can plan on the 2 to 3 PM window. I will send an update if the timing shifts.",
            "I have you down for an arrival between 2 and 3 PM. I appreciate your flexibility.",
            "The confirmed window is 2 to 3 PM. I will keep you posted if anything affects it.",
            "I should arrive between 2 and 3 PM. Thanks for checking in.",
        ],
    },
    {
        "family": "complaint_recovery", "category": "complaint",
        "messages": [
            {"role": "customer", "text": "One part of the job was missed, and I am disappointed."},
            {"role": "owner", "text": "I am sorry. I want to understand exactly what was missed."},
            {"role": "customer", "text": "The back section still needs attention."},
        ], "state": {"followup_booked": False, "calendar_access": False},
        "targets": [
            "You are right to point that out, and I am sorry we missed the back section. I will arrange a follow-up once we confirm timing.",
            "I apologize for missing that area. I have noted the back section and will work with you on the next step.",
            "Thanks for being specific. I am sorry it was overlooked, and I will make sure the follow-up addresses that section.",
            "That is our mistake. I understand which area you mean, and I will confirm how we can correct it.",
            "I am sorry the work was incomplete. I will focus the follow-up on the back section rather than make you repeat the whole request.",
            "I hear you. The back section should have been handled, and I will help get a clear correction plan in place.",
        ],
    },
    {
        "family": "complaint_no_details", "category": "complaint",
        "messages": [
            {"role": "customer", "text": "Something was not right after the visit."},
            {"role": "owner", "text": "I am sorry to hear that. I want to make it right."},
            {"role": "customer", "text": "I am not sure how to describe it."},
        ], "state": {"followup_booked": False},
        "targets": [
            "That is okay. What did you notice, and where did it show up? A photo or a short description may help.",
            "No problem. Tell me what looks different or incomplete, even if it is hard to describe precisely.",
            "I understand. Start with what you expected and what you saw instead, and we can narrow it down.",
            "That is enough to start. Which part concerns you most, and when did you first notice it?",
            "You do not have to phrase it perfectly. What is the clearest thing that seems wrong?",
            "I can help sort it out. Tell me what you noticed first and I will ask any follow-up questions.",
        ], "state": {"followup_booked": False},
    },
    {
        "family": "payment_explain", "category": "payment",
        "messages": [
            {"role": "customer", "text": "I do not understand this invoice."},
            {"role": "owner", "text": "I can walk through it with you."},
            {"role": "customer", "text": "The total is higher than I expected."},
        ], "state": {"invoice_details": "not provided", "payment_processed": False},
        "targets": [
            "I understand. Which line or part of the total looks different from what you expected?",
            "I can explain it, but I do not want to guess without seeing which charge concerns you.",
            "Let us go through it clearly. What line item would you like me to explain first?",
            "Thanks for flagging it. I need to know which part looks off before I can give you an accurate explanation.",
            "I am happy to review it. Tell me which charge you are questioning and I will address that specifically.",
            "I understand the concern. I do not have the invoice details in this message, so which line should we start with?",
        ],
    },
    {
        "family": "payment_boundary", "category": "boundary",
        "messages": [
            {"role": "customer", "text": "Can I send payment through an unusual app instead?"},
            {"role": "owner", "text": "I want to keep payment details clear and secure."},
            {"role": "customer", "text": "It would be easier for me."},
        ], "state": {"accepted_payment_methods": "not provided", "payment_processed": False},
        "targets": [
            "I understand, but I do not want to accept a method that is not confirmed. I can explain the ordinary options we use.",
            "I would rather keep the payment method documented and secure. Let me confirm which options are available.",
            "I understand the convenience, but I cannot approve an unconfirmed payment method. I will clarify the standard options.",
            "For both of us, it is best to use a documented method. I will tell you what is currently accepted.",
            "I cannot promise that app is supported. Let me verify the regular payment options before you send anything.",
            "I hear you. I do not want you to send payment until we have confirmed an accepted method.",
        ],
    },
    {
        "family": "access_safe", "category": "access",
        "messages": [
            {"role": "customer", "text": "I will not be home during the visit. What should I do?"},
            {"role": "owner", "text": "We can keep the access plan simple and safe."},
            {"role": "customer", "text": "I do not want to share anything sensitive by text."},
        ], "state": {"access_instructions": "not provided", "appointment_confirmed": False},
        "targets": [
            "That is sensible. Please do not send sensitive access details by text; we can confirm a safe plan separately.",
            "I agree. Keep private codes and keys out of text messages, and we can discuss an ordinary access option.",
            "No problem. Do not send anything sensitive here. We will confirm a safe way to handle access before the visit.",
            "That is the right precaution. I will not ask you to put a private code in a text message.",
            "We can work out a simple plan without sharing sensitive information here. I will confirm the details first.",
            "Please keep private access information off SMS. We can arrange a safer option once the appointment is confirmed.",
        ],
    },
    {
        "family": "completion_next", "category": "completion",
        "messages": [
            {"role": "customer", "text": "Thanks, the work looks good. Is there anything else I should know?"},
            {"role": "owner", "text": "I am glad it looks good."},
            {"role": "customer", "text": "I just want to make sure I do not miss a next step."},
        ], "state": {"job_status": "complete", "next_step": "not provided"},
        "targets": [
            "You are welcome. There is no special next step unless you notice a concern; just let me know if anything changes.",
            "You are welcome. Keep an eye on it as usual, and reach out if you notice anything that needs attention.",
            "Glad to hear it. Nothing else is required right now, but I am happy to answer a follow-up question.",
            "You are all set for now. If anything seems off later, send me a note and we can look at it.",
            "I appreciate that. There is nothing more you need to do unless a new issue comes up.",
            "You are welcome. The practical next step is simply to let me know if you notice a problem.",
        ],
    },
    {
        "family": "scope_boundary", "category": "boundary",
        "messages": [
            {"role": "customer", "text": "Can you also handle a job outside your normal service?"},
            {"role": "owner", "text": "I want to be clear about what I can safely take on."},
            {"role": "customer", "text": "It is okay if that is not something you do."},
        ], "state": {"requested_work": "outside normal scope", "referral": "not provided"},
        "targets": [
            "That is outside my normal scope, so I do not want to promise work I cannot do well. A specialist would be the safer option.",
            "I appreciate that. I cannot take on that type of work, but I can help clarify what kind of specialist to look for.",
            "I would rather give you an honest no than do work outside my area. That is best handled by the appropriate specialist.",
            "That is not a service I provide, so I cannot schedule it. I hope you find the right professional for it.",
            "I cannot safely take that on. It would be better to use someone whose normal scope includes that work.",
            "Thanks for understanding. I do not want to overpromise outside my scope; a specialist should handle that request.",
        ],
    },
    {
        "family": "clarification", "category": "clarification",
        "messages": [
            {"role": "customer", "text": "Can you take care of it soon?"},
            {"role": "owner", "text": "I want to make sure I understand what you mean by it."},
            {"role": "customer", "text": "The thing we discussed earlier."},
        ], "state": {"request_reference": "ambiguous", "calendar_access": False},
        "targets": [
            "I want to help, but I am not sure which part you mean. Could you name the work you want handled?",
            "Can you remind me which item you are referring to? I do not want to make the wrong assumption.",
            "I may be missing the reference. What specifically would you like me to take care of?",
            "I want to get this right. Which service or issue are you referring to?",
            "Could you clarify what ‘it’ means here? Once I know that, I can explain the next step.",
            "I do not want to guess. Please tell me which part of the earlier conversation you mean.",
        ],
    },
    {
        "family": "repeat_customer", "category": "relationship",
        "messages": [
            {"role": "customer", "text": "The last visit went well. Can we do something similar?"},
            {"role": "owner", "text": "I am glad it worked well. I can use the same general plan if the scope is unchanged."},
            {"role": "customer", "text": "Yes, nothing has changed."},
        ], "state": {"prior_scope": "unchanged", "calendar_access": False},
        "targets": [
            "That sounds good. I will keep the same general scope, and we can confirm timing separately.",
            "Great, I can follow the same general plan. I do not have a confirmed time yet, but we can sort that out next.",
            "I appreciate that. We can use the previous scope as a starting point and confirm the appointment details.",
            "Perfect. I will keep the approach consistent with last time, then we can discuss when it works.",
            "That should be straightforward. I will use the prior scope as a reference without assuming a date.",
            "Glad to hear it. We can keep the same general plan and handle scheduling once the timing is checked.",
        ],
    },
]

CASUAL_CASES = [
    ("That was a really good dinner. I am still thinking about it.", "I am glad you enjoyed it. What was your favorite part?"),
    ("The rain finally stopped here.", "It did here too. Hopefully it stays clear for a bit."),
    ("I saw a tiny frog on the porch this morning.", "That is a fun surprise. I hope it found a dry place to hide."),
    ("I have been trying to learn the guitar, but my fingers are sore.", "That takes patience. Short, regular practice may be easier than one long session."),
    ("My dog is convinced every delivery is for him.", "That sounds about right. He probably thinks the whole neighborhood is sending him gifts."),
    ("I am running late, but I am okay.", "Thanks for letting me know. Take your time and stay safe."),
    ("What is your favorite season?", "Probably fall. I like the cooler air and the change in colors."),
    ("I need a quick distraction from a long day.", "Here is one: a group of flamingos is called a flamboyance."),
    ("I found an old photo from college and it made me laugh.", "Those unexpected memories can make an ordinary day much better."),
    ("I am trying to decide whether to read or watch something tonight.", "A short book is a good choice if you want a quieter evening; a show is easier when you are tired."),
    ("The neighbor's cat keeps sitting on my doorstep.", "It may have decided your doorstep is the best seat in the neighborhood."),
    ("I finally finished a puzzle that took me all week.", "That is satisfying. The last few pieces always seem to take the longest."),
    ("Do you think early mornings are worth it?", "Sometimes. They can be peaceful, but only if you are getting enough sleep."),
    ("My plant has a new leaf.", "That is a good sign. It sounds like it is happy with the light."),
    ("I am learning to bake bread.", "That is a fun project. The dough usually rewards patience more than speed."),
    ("The sunset was pink and orange tonight.", "That sounds beautiful. Those colors make it worth pausing for a minute."),
    ("I forgot why I walked into the room.", "That happens to everyone. Maybe the room will remind you when you walk back out."),
    ("I need a simple idea for lunch.", "A sandwich, fruit, and something crunchy is simple and usually dependable."),
    ("My friend sent me a very long voice note.", "That is a lot of conversation in one message. Hopefully it was a good story."),
    ("I am trying to spend less time on my phone.", "Small boundaries can help, like keeping it away during meals or before bed."),
    ("The library had a book sale today.", "That is dangerous in the best way. It is hard to leave with only one book."),
    ("I walked farther than I planned and now my feet hurt.", "That sounds like a sign to rest. At least the walk was productive."),
    ("What is a small thing that can improve a rough day?", "A good meal, a short walk, or a kind conversation can make a rough day feel lighter."),
    ("I keep putting off cleaning out one drawer.", "Start with five minutes. A small beginning is often enough to make the rest easier."),
    ("The dog fell asleep with a toy in his mouth.", "That is adorable. He must have wanted to keep it close even in his dreams."),
    ("I am excited about a quiet weekend.", "That sounds well deserved. Quiet plans can be the best plans sometimes."),
]

GENERAL_CASES = [
    ("What is a palindrome? Give one short example.", "A palindrome reads the same forward and backward. ‘Radar’ is one example."),
    ("Why does the moon appear to change shape?", "We see different portions of the moon's sunlit half as it orbits Earth."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Give me three ideas for a quiet weekend.", "You could take a walk, cook something new, or spend an hour with a book or podcast."),
    ("Summarize this idea in one sentence: practice helps a skill become more reliable.", "Regular practice makes a skill more consistent and dependable."),
    ("Tell me a two-sentence story about a paper boat.", "A paper boat drifted into a puddle and discovered a reflection of the whole sky. When the wind came, it sailed home."),
    ("What is the difference between weather and climate?", "Weather is the short-term condition of the atmosphere; climate is the long-term pattern in a place."),
    ("I changed topics. Can you explain what a metaphor is?", "A metaphor describes one thing by saying it is another, to highlight a shared quality."),
]


DIVERSE_BUSINESS_DOMAINS = [
    "landscaping", "appliance_repair", "bookkeeping", "pet_grooming",
    "tutoring", "photography", "catering", "salon", "property_management",
    "it_support", "fitness_coaching", "auto_detailing", "painting",
    "roofing", "event_rental", "home_organization",
]


def record(example_id: str, family: str, category: str, messages: list[dict[str, str]], target: str, state: dict[str, Any], mode: str = "business_reply") -> dict[str, Any]:
    return {
        "example_id": example_id,
        "context_group": family,
        "category": category,
        "mode": mode,
        "verified_state": state,
        "messages": messages,
        "target": target,
        "source": "fictional_authored",
        "prompt_variant": variant_for_key(example_id).value,
    }


def build_records(seed: int = 20260806) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(BUSINESS_CASES):
        order = list(range(len(case["targets"])))
        rng.shuffle(order)
        state = dict(case["state"])
        state["business_domain"] = DIVERSE_BUSINESS_DOMAINS[case_index % len(DIVERSE_BUSINESS_DOMAINS)]
        for variant in order:
            rows.append(record(
                f"robust-{case['family']}-{variant:02d}", case["family"], case["category"],
                case["messages"], case["targets"][variant], state,
            ))
    for family, (customer, target) in enumerate(CASUAL_CASES):
        rows.append(record(f"casual-{family:02d}", "casual_sms", "casual", [{"role": "customer", "text": customer}], target, {}, "casual_sms"))
    for family, (customer, target) in enumerate(GENERAL_CASES):
        rows.append(record(f"general-{family:02d}", "general_question", "general", [{"role": "customer", "text": customer}], target, {}, "general_question"))
    return rows


def split_rows(rows: list[dict[str, Any]], eval_mod: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, evaluation = [], []
    for row in rows:
        digest = hashlib.sha256(row["example_id"].encode()).digest()
        (evaluation if int.from_bytes(digest[:2], "big") % eval_mod == 0 else train).append(row)
    return train, evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Build grounded fictional voice data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    rows = build_records(args.seed)
    train, evaluation = split_rows(rows)
    replay = [row for row in rows if row["mode"] in {"casual_sms", "general_question"}]
    for name, values in (("train.jsonl", train), ("eval.jsonl", evaluation), ("all.jsonl", rows), ("replay.jsonl", replay)):
        with (args.output_dir / name).open("w", encoding="utf-8") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "completed", "policy": "fictional_authored_visible_state_only",
        "seed": args.seed, "records": len(rows), "train": len(train), "eval": len(evaluation),
        "business": sum(row["mode"] == "business_reply" for row in rows),
        "casual": sum(row["mode"] == "casual_sms" for row in rows),
        "general": sum(row["mode"] == "general_question" for row in rows),
        "context_groups": len({row["context_group"] for row in rows}),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
