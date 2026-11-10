# BimaCheck AI — Technical Implementation

*AI-powered voice-based insurance policy verification using LiveKit, Gnani, Gemini, and deterministic policy validation.*

## System overview

```text
Customer speech → LiveKit AgentSession → Gnani Prisma STT → Gemini language understanding
                → Python policy checks and workflow → Gnani Timbre TTS → LiveKit audio
```

Gemini interprets customer language and proposes wording or actions. **Python alone owns the current question, state, evidence, policy comparisons, discrepancies, safety decisions, and verdict.** Gemini cannot pass a check or advance the workflow; factual results are compared with the configured policy record.

## Technology stack

- Python 3.11 (target 3.11.9); `livekit-agents==1.8.10`, `livekit-plugins-gnani==1.8.10`.
- LiveKit AgentSession; Gnani streaming STT (`hi-IN`) and Timbre TTS (`timbre-v2.5`, `hi-IN`, WebSocket).
- LiveKit Inference Gemini `google/gemini-3.5-flash-lite`, low reasoning, 8-second timeout, no inference retry.
- JSON policy record, deterministic Python evaluators, pytest.

## Conversation and verification

After consent, Python matches the customer’s name, discloses and confirms the policy facts, then asks four grouped answer turns. The seven policy checks remain separate and are evaluated in this order:

1. Annual premium + premium payment period
2. Policy term + product type
3. Payout recipient + pension payment timing
4. Consequences of stopping premiums

Each answer is checked independently for its adjacent checks. Correct passes and advances; wrong, unclear, or refused answers become discrepancies and advance without a second substantive policy-check attempt. In normal completion, the verdict follows all seven checks; explicit end-call and terminal safety/identity outcomes may finalize earlier. Compatible fragments may be combined before interpretation with a **0.4-second** maximum wait; unrelated turns are buffered, and explicit end-call language is not merged.

| Policy fact | Current record |
|---|---|
| Product | Suraksha Pension Plus; deferred annuity |
| Annual premium / payment period | ₹1,50,000 / 10 years |
| Total policy term | 20 years |
| Payout recipient | Sunita |
| Payout timing | Starts after 5 years |
| Stopping premiums | Policy may lapse or benefits may reduce |

The 5-year payout start is a separate timing value from the 10-year premium payment period. The approved explanation and spoken reference now state that pension payment starts after 5 years.

## Identity, privacy, and deterministic checks

Python matches the customer name and explicit `name_variants` against the original transcript. A match advances directly to policy disclosure without a separate identity confirmation. The expected name context is withheld from Gemini until verification; the demo uses `Vansh` / `वंश`. Matching is case-insensitive substring matching, not fuzzy phonetics. No father’s name is used.

Numeric checks normalize Indian comma grouping, currency-prefixed/plain digits, and supported Hindi/Hinglish number words (for example `₹1,50,000`, `150000`, `डेढ़ लाख`). Python selects an unambiguous number relevant to the pending check and compares it with the record. Recipient matching uses `Sunita` and whole-name alias `सुनीता`. Gemini’s candidate is interpretation only; evaluation uses the original transcript.

## Gemini boundary and safeguards

The prompt/schema describes ten fields; the parser requires non-empty `spoken_response` and valid `intent`, defaults other fields, ignores extra top-level fields, and rejects invalid enums, types, or tool shapes. Parse/provider failures return a safe clarification fallback without another model call or retry. Python pins the check ID and original transcript; Gemini’s candidate is not verification evidence. A proposed handoff cannot override a current mandatory answer. End-call requires explicit customer language.

Python checks risk signals against approved patterns. Verdicts are **Green / `clear`** for all-pass results without review concerns, **Amber / `pending_review`** for discrepancies or non-critical review conditions, and **Red / `hold`** for refused consent or confirmed guarantee/sign-pressure risks.

## Observability and validation

Monotonic logs cover final STT, Gemini request/response, Python work/evaluation, TTS request/generation start, and playback completion/interruption. `TOTAL_TO_FIRST_AUDIO_MS` is `unavailable`; first-audio onset is not reliably measured.

**Current tests:** full suite **153 passed**; focused voice/policy/compliance suite **80 passed**.

## Key implementation improvements

Deterministic identity routing; Python-owned check state and verdicts; suppression of Gemini-directed transitions; corrected handoff precedence; Hindi/Indian-number and recipient matching; four-turn aggregation with bounded waiting; one-pass discrepancy handling; tolerant Gemini response parsing; and turn-level latency instrumentation.
