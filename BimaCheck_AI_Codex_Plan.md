# BimaCheck AI: Build Plan for Codex

A Hinglish voice agent that runs the pre-issuance verification call for life insurance and checks whether the customer understood what they bought, instead of collecting a "yes".

Built by two people: Vansh (voice agent) and a teammate (API, database, dashboard). This plan is for Codex. Follow the architecture, folder ownership and rules below, and build only what is described here. Where the plan says to measure or test something, do that before changing settings or claiming an improvement. The database schema and the data-channel event JSON are the shared contract and change only by agreement between both people.

---

## 1. What we are building

**One sentence:** an automated assistant calls a (synthetic) insurance buyer, asks open questions in Hinglish, checks the answers against the policy record, and gives each call a Green, Amber or Red verdict. Red and Amber calls go to a human review queue with quoted evidence.

**What makes it different:** many automated verification flows read a script and match the reply. BimaCheck AI asks the customer to explain things back, and judges understanding. It also listens for mis-selling signals such as "FD hai", "bas sign kar do", or a family member answering for the customer.

**Non-goals (to keep it small):**
- No real customers, policies, phone numbers or recordings. Everything is synthetic.
- No claim of IRDAI compliance. The agent flags calls for a human, it never approves or rejects.
- No phone/SIP layer, no multi-agent setup, no RAG, no message queue, no Postgres.

## 2. Constraints that shape the build

- The app must genuinely use Gnani tech: Prisma (speech to text) and Timbre v2.5 (text to speech) via the API platform at app.gnani.ai/voice. Stay on Timbre v2.5 unless evidence from the voice-quality test set (section 8.2) says otherwise.
- Report only measured numbers and label everything synthetic. No performance claim without a measurement behind it.
- 5,000 API credits only. Cached audio and short test calls protect them.

## 3. Call behaviour

### 3.1 Stages

1. **Connect:** customer joins from the browser.
2. **Disclosure and consent:** the agent says it is an automated assistant and that the call is recorded, and asks to proceed. If consent is refused, the call ends and is routed to a human.
3. **Identity:** confirms the customer against synthetic details.
4. **Fact confirmation:** plan name, premium amount, premium term, policy term.
5. **Comprehension checks:** open questions on the points that cause mis-selling (see 3.2).
6. **Wrap up:** thanks the customer and states what happens next (human review or proceed).

### 3.2 The checks

Each check is one question with an expected answer taken from the policy record.

| Key | Question idea | Mandatory | Evaluated by |
|---|---|---|---|
| premium_amount | How much premium, in your own words | yes | code |
| premium_term | For how many years do you pay | yes | code |
| policy_term | How long does the policy run | yes | code |
| product_type | Is this an FD or something else | yes | judge |
| payout_recipient | Who receives the money | yes | judge |
| payout_timing | When does the money start | yes | code or judge |
| stop_paying | What happens if you stop paying | yes | judge |
| affordability | Premium against declared income (seniors) | no, warning only | code |

Each check follows the same loop: ask, listen, evaluate. If the answer is wrong or unclear, the agent gives one pre-written re-explanation, then asks once more. If it is still wrong, the check fails.

### 3.3 Flags (always listening)

Flags are detected by plain code on every customer turn, in parallel with the checks.

- **FD_CONFUSION:** customer calls it an FD, fixed deposit, or "ek hi baar".
- **GUARANTEE_CLAIM:** "guaranteed", "pakka return", "bank wale ne bola".
- **SIGN_PRESSURE:** "bas sign kar do", "bete ne bola", "agent ne bola kar do".
- **THIRD_PARTY:** someone else is clearly answering. The agent asks to speak to the policyholder. If unresolved, the call ends with a flag.

The keyword lists live in one file and must cover both Roman and Devanagari output, since we do not yet know which script Prisma returns for Hinglish. The stress test settles this.

### 3.4 Example (Red scenario)

Synthetic record: Ramesh Kumar, 72, declared income Rs 6 lakh a year. Product: Suraksha Pension Plus, a deferred annuity, not an FD. Premium Rs 1.5 lakh per year for 5 years. Pension starts after a 5-year deferment and goes to his daughter Sunita.

> Agent: Aapki policy ka premium kitna hai, aur kitne saal bharna hai? Apne shabdon mein bata dijiye.
> Customer: Dedh lakh. Bank wale ne bola ek baar mein ho jayega.
> Agent: Dedh lakh sahi hai. Lekin record ke mutabiq yeh har saal, paanch saal tak bharna hai. Ek baar phir bata dijiye, kitne saal premium dena hai?
> Customer: Ek hi baar na? FD jaisa hai na?
> Agent: Yeh FD nahi hai, yeh pension plan hai. Pension ka paisa kisko milega, kab se?
> Customer: Mujhe milega na. Bete ne bola bas sign kar do.
> Agent: Dhanyavaad. Main yeh call ek senior colleague ko bhej raha hoon. Tab tak policy issue nahi hogi.

Result: premium amount passes, premium term fails, product type fails, recipient fails, flags FD_CONFUSION, GUARANTEE_CLAIM and SIGN_PRESSURE. Verdict Red, issuance on hold in the demo database.

Amber example: the customer says "bees saal, ya paachis?" on the first try, gets the pre-written re-explanation, then answers correctly. That is Amber with a "needed re-explain" note.

## 4. System architecture

```
Browser (web/, React + Vite)                    LiveKit Cloud
 - call screen: mic, live checklist  <--------> room <--------> Agent worker (agent/)
 - review queue, call detail, report                              |
        ^   ^                                                     |-- Prisma STT (Gnani plugin)
        |   +-- LiveKit data channel (checklist, transcript, -----|-- turn engine (plain Python)
        |       flags, verdict) <---------------------------------|     state machine, number parser,
        |                                                         |     flag detectors
        |                                                         |-- LLM judge (ambiguous free text only)
        v                                                         |-- Timbre TTS + pre-generated audio
  API (api/, small FastAPI)                                       |-- policy held in memory per call
   POST /token  GET /policies  GET /calls                         |-- audio capture + latency logger
   GET /calls/{id}  POST /calls/{id}/review-action                |
   GET /audio/{call_id}                                           v single writer thread
        ^                                                   SQLite (WAL, busy timeout) + audio/
        +------------ reads / review writes ------------------------^
```

**Two paths, kept apart:**
- **Realtime path:** browser, LiveKit, agent. No FastAPI, no database reads, no RAG. This is the only path that affects how the agent feels.
- **Async path:** database writes, final verdict, optional audit, review queue, report. Nothing here can slow the call.

## 5. Components

### 5.1 Agent (agent/)

Python, LiveKit Agents with Gnani's LiveKit plugin (check docs.gnani.ai for the exact package name and options).

- **Startup:** reads call metadata (call_id, policy_id, scenario) from the room, loads the policy and customer from SQLite once into memory, and never touches the database for reads again.
- **State machine:** owns the stage and check order. It cannot skip a mandatory check, and it decides the next question. The LLM never controls flow.
- **Number parser:** converts Hindi and mixed number words to integers: ek, do, teen, paanch, bees, paintalis, dedh (1.5), dhai (2.5), sawa, hazaar, lakh, crore, plus plain digits. Built with a table of test cases, because it is the most failure-prone and most measurable part.
- **Flag detectors:** keyword and regex rules from one config file.
- **LLM judge:** called only for free-text checks (product_type, payout_recipient, stop_paying, some payout_timing).
  - Input: the question, the expected facts, the customer's answer.
  - Output: strict JSON `{"verdict": "correct|incorrect|unclear", "quote": "...", "reason": "..."}`.
  - Temperature 0, short prompt, small fast model, hard timeout around 1.5 seconds. A timeout counts as "unclear".
  - "Unclear" never becomes Green. It triggers one re-ask, and if still unclear the check is Amber and goes to review.
- **Speech output:** all scripted lines, including re-explanations, are pre-written and pre-generated with Timbre at seed time and stored in an audio cache. The agent never improvises policy terms, which removes a risk and saves latency. Short backchannels ("ji", "achha") are cached too and played while the judge runs. Every string sent to Timbre, cached or generated at call time, passes through the single normalization function described in section 8.2.
- **Barge-in:** agent audio stops the moment the customer starts speaking.
- **Warm-up:** at worker start (prewarm) import heavy modules, load models such as the VAD, load the audio cache into memory, and open connections to the Gnani STT and TTS endpoints and the LLM provider, so the first customer turn does not pay first-use costs. At call start, before the greeting finishes, exercise STT, TTS and the judge once with a minimal request. Nothing blocking (imports, file or SQLite access, SSL setup, synchronous HTTP) may run on the event loop during a turn. Keep any warm-up step only if the cold versus warm measurement in section 7.1 shows it helps.
- **Turn commit and late transcripts:** the agent must handle a transcript that arrives after LiveKit has already committed the customer's turn, instead of ignoring it. Full requirements are in section 7.2. Inspect LiveKit's turn-handling events and options to see how to detect this, and do not guess their behaviour.
- **Latency logger:** records the four latency metrics in section 7.1 separately for every turn, with turn type and cold or warm status, into turn_metrics.
- **Failure handling:** if STT, the LLM or TTS fails, the agent asks the customer to repeat. After two failures it offers a human callback and marks the call needs_human. It never goes silent.
- **Audio capture:** the customer's audio is saved to audio/ with offsets, so each check can be linked to a snippet.
- **Writer thread:** a single background thread receives events from a queue and writes to SQLite, so a blocking write never stalls the call.

### 5.2 API (api/)

A small FastAPI service, one file to start. It is not in the call path.

| Endpoint | Purpose |
|---|---|
| POST /token | creates a call row, returns a LiveKit join token (keeps the secret off the browser) |
| GET /policies | lists synthetic policies and scenarios for the call screen |
| GET /calls | review queue, filter by verdict and status |
| GET /calls/{id} | full call detail: checks, flags, verdict, timings |
| POST /calls/{id}/review-action | callback, release, escalate, with a note |
| GET /audio/{call_id} | serves the saved audio |

No auth, no background jobs. Add nothing else until the engine works.

### 5.3 Web (web/)

React with Vite, Tailwind, and LiveKit's React components. One flat folder.

- **Call screen:** scenario picker, start button, live waveform and transcript, checklist that turns green or red from data-channel events, and a verdict card at the end.
- **Review queue:** table of calls with verdict, status and time.
- **Call detail and report:** per-check table (question, expected, heard, verdict, confidence), quotes, audio snippet player, flags, reviewer actions. Styled for print so "save as PDF" gives the report.

### 5.4 Data channel events

Small JSON messages from the agent to the browser:

```json
{"type": "transcript", "role": "customer", "text": "...", "final": true, "ts_ms": 12840}
{"type": "check_update", "key": "premium_term", "verdict": "fail", "heard": "ek baar", "ts_ms": 19300}
{"type": "flag", "flag": "FD_CONFUSION", "quote": "FD jaisa hai na", "ts_ms": 20100}
{"type": "verdict", "verdict": "red", "reasons": ["premium_term", "product_type", "payout_recipient"]}
```

A transcript event that arrives after its turn was committed carries `"late": true`. This is a contract change and must be agreed with the teammate and added to `db/events.md`.

### 5.5 Database (SQLite, WAL mode, busy timeout set)

Plain flat schema so a later move to a hosted database is a copy job.

- **policies:** id, product_name, product_type, premium_amount, premium_term_years, policy_term_years, payout_start_after_years, payout_recipient, plus the Hinglish explanation text per fact.
- **customers:** id, name, age, declared_income, scenario_label.
- **calls:** id, policy_id, customer_id, started_at, ended_at, status (in_progress, complete, needs_human, no_consent), verdict, hold_status, audio_path.
- **checks:** id, call_id, check_key, expected, heard, verdict (pass, fail, unclear, pass_after_retry), retries, stt_confidence, judge_used, start_ms, end_ms.
- **flags:** id, call_id, flag_type, quote, ts_ms.
- **review_actions:** id, call_id, action, note, created_at.
- **turn_metrics:** id, call_id, turn_no, turn_type (fixed, judge, generated), is_cold, end_of_speech_ms, turn_committed_ms, stt_final_ms, llm_first_token_ms, tts_first_audio_ms, total_ms, late_transcript (0 or 1), late_transcript_text. The last four additions (turn_type, is_cold, turn_committed_ms, late_transcript fields) are a schema change and must be agreed with the teammate and added to `db/schema.sql`.

## 6. Verdict rules (Python, not the LLM)

Applied after the call from the rows above.

- **Red:** any mandatory check still failed after the re-explain, or any of FD_CONFUSION, GUARANTEE_CLAIM, SIGN_PRESSURE, or THIRD_PARTY unresolved, or consent refused.
- **Amber:** no Red conditions, but at least one check passed only after a re-explain, had low STT confidence, or was left "unclear".
- **Green:** every mandatory check passed first time, no flags, no low-confidence turns.

Hold status follows the verdict: Red sets hold, Amber sets pending review, Green sets clear. The hold is a status in the demo database, and it is written by code only after the real change. The UI never fakes it.

An optional post-call LLM audit re-reads the transcript and adds notes. It may escalate a verdict but never relax one.

**Unclear or unresolved answers never receive Green.** This holds in every code path, including judge timeouts, STT, LLM or TTS failures, low STT confidence, and late or partial transcripts. A check whose answer arrived as a late or partial transcript and was not re-evaluated counts as unclear.

## 7. Latency and turn detection

### 7.1 Latency

Latency in a cascaded agent is the gap from the customer finishing a sentence to the first audio back. Most of it can be removed by not using the LLM.

**Design (unchanged):**
- Fixed lines and re-explanations play from cached audio, with no LLM and no TTS at call time.
- Numbers and flags are handled in code.
- The LLM runs only on ambiguous free-text answers, with a backchannel covering the pause.
- Pass STT partials forward, and keep the agent, the LLM provider and the Gnani endpoints in nearby regions.
- Use browser WebRTC, since telephony adds its own delay.

**Measure four things separately, for every turn:**
1. **STT latency:** end of customer speech to final transcript.
2. **LLM time to first token:** judge turns only.
3. **TTS time to first audio:** for any line generated at call time. Cached lines record playback start separately and are labelled cached.
4. **End-to-end response latency:** end of customer speech to first agent audio.

Log them in turn_metrics with turn_type and is_cold. Report p50 and p95 for each metric, split by turn type (fixed, judge, generated) and by cold (first turn after worker start) versus warm.

**Find and fix the cold-start causes, with measurements:**
- **First-use imports and event-loop blocking:** do imports and model loading in prewarm, and keep blocking work off the event loop. Measure event-loop lag during the first turns to confirm.
- **SSL initialization delays:** open and reuse connections to Gnani STT, Timbre and the LLM provider before the first customer turn. Measure first-request versus later-request time for each provider.
- **Slow audio generation:** any line generated at call time is measured. Scripted lines must come from the cache, and a slow generated line should be moved into the cache.
- **Warm-up:** see the warm-up requirement in section 5.1. Record the cold versus warm gap before and after.

**No claims without measurement:** targets are set only after a baseline exists. Any statement that latency improved needs a before and after measurement stored in `eval/`.

### 7.2 Turn detection

**Observed problem:** in LiveKit with Gnani Prisma STT, a transcript arrived after LiveKit had already committed the user's turn. For customers who pause mid-sentence, this can cut them off or lose words.

**Requirements:**
1. Test slow, elderly-style speech with mid-sentence pauses, and noisy audio, before choosing endpointing settings.
2. Tune endpointing from evidence. Too aggressive and the agent interrupts, too conservative and it lags. For each setting tried, record whether turns were cut off, how many late transcripts occurred, and the added latency. Keep the chosen settings and the evidence in `eval/`.
3. Late or partial transcripts are never silently lost. When a transcript arrives after the turn was committed, the agent must: log it (turn_metrics.late_transcript and late_transcript_text), forward it to the browser with `"late": true`, and re-evaluate the affected check with the full text. If re-evaluation is not possible, the check is "unclear" and the agent re-asks once. A late transcript never produces Green by itself.
4. Barge-in stays as specified: agent audio stops the moment the customer starts speaking.

## 8. Synthetic data and voice-quality test set

### 8.1 Synthetic data

- **Primary product:** Suraksha Pension Plus, a deferred annuity. Add a second product (a simple endowment) only if time allows.
- **Scenarios:** (1) confused senior who thinks it is an FD (Red), (2) hesitant but corrects after re-explain (Amber), (3) informed and correct (Green), (4) family member answers (THIRD_PARTY flag), (5) noisy line with low STT confidence, (6) pressure phrase "bete ne bola sign kar do".
- **Names and numbers:** clearly fictional. Pre-generate the audio cache with a seed script (make seed).

### 8.2 Hinglish voice-quality (pronunciation) test set

A repeatable test of how Timbre v2.5 pronounces our actual content, so wrong wording, names or numbers are caught by listening, not discovered on a call.

**Location:** `eval/pronunciation/`. A lines file with fields id, category, source_text, normalized_text and notes. A script that renders every line with Timbre v2.5 into an audio folder. A results file with a listener verdict (ok, wrong, awkward) and notes per line.

**Coverage:**
- **Insurance terms:** premium, policy, annuity, deferred, maturity, nominee, surrender, lock-in, sum assured, free-look period, pension, FD.
- **Names:** the fictional customers and nominees, the product name (Suraksha Pension Plus) and the assistant name (BimaCheck AI).
- **Numbers:** amounts (Rs 1.5 lakh, Rs 7.5 lakh, Rs 6 lakh), years, ages, percentages, and spoken forms such as "dedh lakh" and "saat lakh pachaas hazaar".
- **Premiums and durations:** phrases such as "har saal, paanch saal tak" and "bees saal premium, paachis saal policy", plus deferment periods.
- **Real script lines:** every disclosure, check question, re-explanation, backchannel and fallback line exactly as it will be spoken. Where useful, test Roman and Devanagari versions of the same line.

**Script and wording normalization:** one function that turns script text into TTS text. It decides the script per segment, writes numbers and units in the form that sounds right, expands abbreviations (FD, Rs), and applies a small replacement table for known mispronunciations. All TTS input, cached or generated at call time, goes through it. It has transcript-level tests, and the replacement table grows from listening notes.

**Listening is part of the test:** a person listens to every generated clip and records a verdict. No automated pass. Only lines marked ok go into the audio cache. When the normalization function changes, regenerate and re-listen to the affected lines.

**Provider and model:** keep Timbre v2.5. Try other Timbre v2.5 voices and different wording first. Changing model or provider needs recorded evidence from this set that the problem cannot be fixed with normalization or voice choice.


## 9. Evaluation

Two levels, both small and honest about sample size.

**A. Transcript-level tests (fast, deterministic, run on every change):**
- Number parser: 60 or more utterances including dedh, dhai, sawa, mixed digit and word forms.
- Flag detectors: positive and negative examples in Roman and Devanagari.
- Judge: 30 labelled answers across the free-text checks.
- Normalization function: tests for numbers, units, abbreviations and the replacement table.
- Late and partial transcripts: simulate a transcript arriving after the turn was committed and confirm it is logged, forwarded with `"late": true`, re-evaluated, and never ends in Green when unresolved.
- Verdict safety: judge timeout, STT, LLM or TTS failure, low confidence and late transcripts all end Amber or Red, never Green.

**B. Live-call tests:**
- 30 or more scripted calls with friends across the six scenarios, recorded with real voices, some with TV or street noise at three levels (play the noise from a speaker).
- Ground truth labelled beforehand: understood or misunderstood.
- Include slow, elderly-style speech with mid-sentence pauses.
- Report: flag recall on misunderstood calls, false-flag rate, number accuracy, STT confidence by noise level, turn cut-offs and late transcripts, and p50/p95 for each of the four latency metrics (cold versus warm).
- Compare Prisma against a generic STT on the same noisy audio, fairly and with the method stated.

**C. Voice-quality test:** the listening test in section 8.2, with results kept in `eval/pronunciation/`.

## 10. Repo layout (flat)

```
bimacheck/
  agent/     [Vansh] LiveKit agent, turn engine, parser, detectors, judge, writer
  api/       [teammate] FastAPI app (single file to start)
  web/       [teammate] React + Vite app
  db/        schema.sql, events.md (data-channel contract), seed script, bimacheck.db, audio_cache/, audio/
  eval/      test utterances, recorded calls, labels, scoring scripts
  README.md
  run.sh     starts API, agent and web with one command
```

## 11. Stack

- Agent: Python, livekit-agents, Gnani LiveKit plugin.
- STT and TTS: Gnani Prisma and Timbre v2.5. Timbre v2.5 stays for now. Do not change provider or model without evidence from the voice-quality test set.
- LLM judge: a small fast model from any provider, chosen after measuring latency and JSON reliability. Evon is optional for the post-call audit only, and only if weights arrive and a GPU is available. Do not claim "sovereign" if the judge is a foreign API.
- API: FastAPI with uvicorn.
- DB: SQLite with WAL.
- Web: React, Vite, Tailwind, LiveKit client and React components.

## 12. Team ownership and working rules

### Roles

| | Vansh | Teammate |
|---|---|---|
| Owns | `agent/`, transcript tests in `eval/` | `api/`, `db/`, `web/` |
| Git branch | `agent` | `dashboard` |
| Builds | voice engine, number parser, flag detectors, judge, latency logging, audio capture, data-channel events | seed data, FastAPI, call screen, review queue, call detail and report, fake event emitter |
| Shared | the contract, live-call eval, demo recording, the post | |

The dashboard only needs to be clean and correct, and it can be built against fake data and the fake event emitter, so neither person waits on the other.

### Working rules

- **Freeze the contract on day 1:** `db/schema.sql` and `db/events.md` (the data-channel event JSON from section 5.4). Change either only by agreement, and tell the other person the same day.
- **Two branches, no pull requests:** `agent` and `dashboard`. Merge each into `main` with a plain `git merge` at the milestones below. `main` should always be demo-ready.
- **Authorship stays visible:** commit under your own name and email, only inside your own folders, and run `git pull --rebase` before pushing. Run Codex only inside your own folders and commit manually.
- **README gets a "Who built what" section** listing each person's folders and features.
- **Codex builds only what this plan describes.** No extra features, and no new dependencies or services without agreement.

## 13. Risks

- **Prisma accuracy on elderly speech and Hindi numbers:** the biggest risk. The stress test (gate 1) decides it.
- **Script variance:** Hinglish may come back in Roman or Devanagari. Normalise both.
- **Turn cut-offs and late transcripts:** customers who pause can be cut off. Handled by section 7.2.
- **Cold-start latency:** first-use imports, SSL setup and event-loop blocking can add delay on the first turns. Handled by section 7.1.
- **Pronunciation:** Timbre v2.5 may mispronounce insurance terms, names or numbers. Handled by section 8.2.
- **Credits:** 5,000 total. Measure per-minute usage in the stress test, keep test calls short, rely on cached audio.
- **LLM latency or outage:** hard timeout, treat as unclear, never go silent.
- **Domain accuracy:** this is not an insurance expert's design. Validate the checks with the research notes and a few real agents.
- **Claims:** do not say it prevents mis-selling or is compliant. It flags calls for human review.

## 14. Open decisions

- Which LLM for the judge, after measuring.
- Endpointing settings for slow speakers, decided from the evidence in section 7.2.
- Which Timbre v2.5 voice, decided by listening in section 8.2.
- Whether to use Evon for the audit.
- Whether a second product is worth adding.
- LiveKit recording versus saving audio in the agent (current plan: save in the agent).

## 15. Quality gates and acceptance criteria

Each gate must pass before the work that depends on it is treated as done.

**Gate 1: Stress test (before building the engine).**
- Run Prisma on noisy Hinglish numbers and slow, elderly-style speech.
- Record number accuracy, which script it returns (Roman or Devanagari), STT confidence behaviour, credits used per minute, and whether late transcripts occur.
- A go or no-go decision is recorded in `eval/`.

**Gate 2: Latency (section 7.1).**
- STT latency, LLM time to first token, TTS time to first audio and end-to-end latency are logged separately for every turn.
- p50 and p95 are reported per metric, by turn type and cold versus warm.
- Cold-start causes (imports and event-loop blocking, SSL initialization, slow audio generation, warm-up) are each measured and either fixed or documented.
- No latency claim exists without a before and after measurement.

**Gate 3: Turn detection (section 7.2).**
- Endpointing settings are chosen from tests on slow, elderly-style and noisy audio, with the evidence stored.
- Every late or partial transcript is logged, forwarded with `"late": true` and re-evaluated. None is silently lost.

**Gate 4: Pronunciation (section 8.2).**
- The test set covers insurance terms, names, numbers, premiums, durations and every real script line.
- Every cached clip was listened to and marked ok.
- Timbre v2.5 is still in use, or a change is backed by recorded evidence.

**Gate 5: Verdict safety (section 6).**
- Tests prove unclear or unresolved answers never receive Green in any code path.

**Gate 6: Evaluation (section 9).**
- Transcript-level tests pass and run on every change.
- Live-call results across the six scenarios are scored and reported honestly, including sample size.

**Gate 7: End to end.**
- A full Red call and a full Amber call run by voice without errors.
- The checklist updates live in the browser.
- The review queue shows the call with quoted evidence and audio.
- The report prints to PDF.

**Non-negotiable:** latency measurement, number parser tests, verdict rules, and the Red scenario.
