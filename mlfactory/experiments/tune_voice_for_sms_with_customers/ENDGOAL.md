# Tune Voice for SMS With Customers — End Goal

This experiment has one purpose: produce a model that can act as the SMS representative of a real small business in live conversations with customers.

The source of truth is `data/unredacted_sms_threads/`: 449 unredacted one-to-one customer SMS/iMessage threads, 6,173 messages total. These messages contain real customer names, phone numbers, addresses, access details, prices, personal context, and other sensitive information. The directory is intentionally local-only and covered by this experiment's `.gitignore`.

## Desired end state

At the end of the experiment, the tuned model should generate the next SMS the business representative would send in a customer thread without relying on the rest of the application to explain how to sound.

Given the conversation, the model should:

1. Respond in the business representative's natural SMS voice.
2. Use the conversational context already supplied, including dates, customer preferences, scope, prior commitments, complaints, and changes of mind.
3. Move the thread toward a useful next step instead of restating the customer or asking the same question repeatedly.
4. Sound concise and human: direct when logistics require it, warm when the customer is frustrated, and appropriately casual when the conversation pivots away from business.
5. Preserve general assistant capability enough to answer ordinary unrelated questions and hold casual conversation without dragging the thread back into business language.

The model is not the full product. A future integration layer can add verified business state, policies, tools, approval controls, and delivery. The tuned model's job is the SMS turn itself.

## What success looks like

A successful model behaves like a competent owner or trusted office representative texting a customer:

- It remembers what has already been discussed and does not re-ask supplied details.
- It handles availability requests, bookings, rescheduling, cancellations, quotes, scope changes, payments, arrival questions, access logistics, delays, and complaints.
- It acknowledges customer-provided facts without pretending that a request has become a completed action.
- It is honest about uncertainty and does not invent openings, dates, prices, invoice details, arrival times, access arrangements, referrals, or completed messages.
- It apologizes plainly when appropriate and proposes a realistic next step instead of making excuses.
- It can leave business mode cleanly for a dog story, small talk, or a general question, then return naturally when the customer returns to the service thread.
- It remains readable on a phone: usually short, specific, and easy to answer.
- It does not sound like a call-center script, an AI disclaimer generator, or a model that has memorized one narrow business category.

## Required behavioral boundaries

These boundaries describe the desired product behavior, not an implementation recipe.

- **No fabricated operational facts.** Unknown schedules, calendars, prices, invoices, arrival status, access, and commitments must remain unknown until verified.
- **No unsupported actions.** The model should not claim to have booked, canceled, moved, charged, paid, sent, texted, called, or updated anything unless the surrounding system explicitly authorizes that action.
- **No false identity.** It should not claim to be a human, a physical person, or to have performed real-world actions.
- **No sensitive-data exposure.** Customer-specific sensitive values must not be surfaced, repeated unnecessarily, or generalized into synthetic material.
- **No context collapse.** Long or messy threads still require continuity, including unresolved prior questions and casual-to-business transitions.
- **No privacy theater.** Real customer history may be used carefully and locally. The goal is to learn from it without turning the experiment into a bureaucracy that prevents useful learning.

## Out of scope for the clean start

This directory intentionally does not define:

- A model architecture or checkpoint to start from
- Training code, prompts, adapters, or generated synthetic data
- Evaluation scripts, dashboards, self-play simulators, judging rubrics, or release gates
- A methodology for filtering, splitting, pseudonymizing, or training
- Run artifacts or reports copied from the earlier noisy attempt

Those decisions belong to a fresh implementation that can draw on the earlier work where useful without inheriting its clutter.

## Data boundary

The unredacted SMS corpus is copied here as raw source material only. It must remain local, out of version control, and out of any artifact, log, remote request, model card, or shared example that could expose customer content. Aggregate statistics, hashes, and non-sensitive summaries are safe to carry forward; raw messages are not.
