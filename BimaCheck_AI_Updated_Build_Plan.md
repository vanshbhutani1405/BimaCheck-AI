# BimaCheck AI --- Updated Build Plan

## Demo-first conversational insurance verification agent

**Project objective:** Build a genuine, tool-using conversational voice
AI agent that can deliver a polished, reliable **40--60 second demo** of
insurance verification.

**Primary engineering priorities** 1. Natural, context-aware
Hindi/Hinglish conversation. 2. Reliable verification, compliance
evidence and verdicts. 3. Low end-to-end response latency. 4. A clear
live dashboard showing the agent's actual decisions and actions. 5. A
modular codebase that demonstrates real agent architecture without
unnecessary complexity.

------------------------------------------------------------------------

## 1. Product direction

Build **one LLM-driven conversational agent with real tools and
independent Python guardrails**.

The agent should understand what the customer means, answer questions,
handle interruptions and confusion, decide when to use tools, and resume
pending verification questions naturally.

Python remains responsible for executing business operations, validating
evidence, enforcing mandatory checks and calculating the final verdict.

This is not: - A rigid question-answer IVR with an LLM added only for
selected checks. - An unrestricted chatbot that can make insurance
decisions. - A multi-agent system. - A generic insurance chatbot for
every product.

It is a focused, demo-ready insurance verification agent for one
internally consistent synthetic policy.

## 2. Target architecture

``` text
Customer / Browser
      |
      v
LiveKit voice session (WebRTC)
      |
      v
Gnani Prisma STT
(streaming transcript, Hindi/Hinglish)
      |
      v
Single Conversational LLM Agent
- understands intent and context
- handles questions, confusion, corrections and interruptions
- chooses tools
- generates concise spoken responses
      |
      v
Python Agent Orchestrator
- session context and conversation history
- pending verification question
- tool execution and validation
- retry counts and workflow safeguards
      |
      +------------------------+
      |                        |
      v                        v
Agent Tools              Python Compliance Engine
- record_answer          - mandatory checks
- get_policy_fact        - risk rules
- flag_risk              - evidence validation
- handoff_to_human       - retry rules
- end_call               - final verdict
      |                        |
      +------------+-----------+
                   |
                   v
           Agent response
                   |
                   v
           Gnani Timbre TTS
                   |
                   v
          Customer hears response


Separate application/API layer:
Voice Agent <---- authenticated API / events ----> Backend
                                                   |
                                                   +-- Database
                                                   +-- Dashboard API
                                                   +-- Session/event records
                                                   |
                                                   v
                                             Teammate's Dashboard
```

### Core design principle

**The LLM owns conversation; Python owns execution and compliance.**

The LLM may propose an action or call a tool, but Python validates the
arguments and checks whether the action is permitted in the current
session state. The LLM cannot mark a mandatory check as passed or choose
the final verdict.

------------------------------------------------------------------------

## 3. Voice and language pipeline

Keep the currently tested voice integrations initially:

  -----------------------------------------------------------------------
  Layer                   Initial choice          Responsibility
  ----------------------- ----------------------- -----------------------
  Real-time transport     LiveKit                 Browser audio, session
                                                  lifecycle and media

  Speech-to-text          Gnani Prisma            Customer speech to
                                                  transcript

  Conversation model      Existing Gemini         Intent understanding
                          integration, subject to and response generation
                          evaluation              

  Text-to-speech          Gnani Timbre v2.5       Spoken agent response
  -----------------------------------------------------------------------

Do not switch providers or models until testing demonstrates a specific
limitation.

### Hindi/Hinglish script policy

Use native-script Hindi and preserve English terms in Latin script.

Example:

`जी, मैं आपको आपकी policy के बारे में समझाता हूँ। आपका premium पाँच साल तक जमा होगा।`

Rules: - STT should preserve Devanagari Hindi and English code-mix where
supported. - Pass the original transcript to the LLM; do not add an
unnecessary translation step. - Instruct the LLM to write Hindi words in
Devanagari and English technical/insurance terms in English script. - Do
not write Hindi sentences in Roman transliteration. - Send the generated
response directly to TTS without an unnecessary transliteration or
translation service. - Configure and verify the appropriate Hindi locale
(`hi-IN`) and a voice that supports the required language. - Keep
responses concise and easy to speak. - Speak amounts, durations and
numbers unambiguously.

### Language quality experiment

Before changing code, test the same 10--15 representative Hindi/Hinglish
sentences in: 1. Devanagari Hindi. 2. Romanised Hindi. 3. Devanagari
Hindi with English code-mix.

Use the same TTS model, voice and settings. Compare pronunciation,
clarity and naturalness. Record the exact text sent to TTS.

------------------------------------------------------------------------

## 4. Conversational agent design

Use one primary LLM agent for every meaningful customer turn. Avoid
multiple separate LLM calls for classification, response generation and
tool selection when one structured agent decision can do the work.

The agent receives: - Current customer utterance. - Relevant recent
conversation history. - Current verification context. - Previously
recorded answers and their statuses. - Exact pending question. -
Relevant approved synthetic policy facts. - Available tool definitions
and their constraints.

The agent should recognise at least these conversational intents:

  -----------------------------------------------------------------------
  Intent                              Expected behaviour
  ----------------------------------- -----------------------------------
  `answer`                            Extract the answer and request
                                      verification/recording

  `customer_question`                 Answer using approved policy facts,
                                      then preserve pending question

  `confusion`                         Explain more simply and check
                                      understanding

  `correction`                        Re-evaluate the corrected answer
                                      and preserve evidence history

  `interruption`                      Stop speaking when appropriate and
                                      listen

  `incomplete`                        Wait or ask a concise clarification

  `unrelated`                         Respond briefly and guide the
                                      customer back

  `refusal`                           Respect refusal and follow the
                                      defined escalation/closure rules

  `silence`                           Use a controlled, non-repetitive
                                      prompt or timeout path
  -----------------------------------------------------------------------

### Session context

Maintain structured session state in Python, for example:

``` python
session_context = {
    "session_id": "...",
    "current_stage": "fact_confirmation",
    "pending_question": "premium_term",
    "answers": {},
    "verification_results": {},
    "retry_counts": {},
    "risk_flags": [],
    "consent_status": "granted",
    "conversation_history": [],
    "disclosed_policy_facts": []
}
```

This is illustrative. Adapt the schema to the existing codebase.

The pending question must survive side conversations. If the customer
asks a policy question while a verification question is pending, answer
the question and then return to the exact unanswered question. Do not
advance a mandatory check merely because the customer said something.

### Agent persona and response rules

The agent should sound like a helpful, professional insurance
representative: - Natural, concise Hindi/Hinglish. - One question at a
time. - No robotic repetition. - No unsupported promises or claims. - No
pretending to know information absent from the policy data. - Clear
explanations without overwhelming the customer. - Respectful handling of
confusion, correction and refusal.

------------------------------------------------------------------------

## 5. Agent tools

Start with a small set of real tools.

  -----------------------------------------------------------------------
  Tool                                Responsibility
  ----------------------------------- -----------------------------------
  `record_answer`                     Save the customer's answer,
                                      transcript evidence and relevant
                                      context

  `get_policy_fact`                   Return approved facts from the
                                      synthetic policy configuration

  `flag_risk`                         Record a risk flag with the
                                      supporting customer statement

  `handoff_to_human`                  Mark the session for human
                                      review/escalation

  `end_call`                          Close the session through the
                                      controlled session lifecycle
  -----------------------------------------------------------------------

Tool requirements: - Tools must perform actual operations; no decorative
or fake tool calls. - Validate arguments and permissions in Python. -
Return concise structured results to the agent. - Record tool name,
arguments, result, timestamp and session ID. - Keep local tool
operations fast. - Do not create unnecessary tool-call loops. - A tool
result is not itself a compliance verdict.

The compliance engine may be called internally by the orchestrator or
relevant tools. It must not be directly overridable by the LLM.

------------------------------------------------------------------------

## 6. Verification and compliance

Retain the project's mandatory checks and existing verdict policy unless
an audit identifies a documented defect. Do not weaken compliance rules
to make the demo easier.

Expected verification areas include: - Identity. - Premium amount. -
Premium payment term. - Policy term. - Product type understanding. -
Payout recipient understanding. - Payout timing understanding. -
Consequences of stopping premium payments.

Keep deterministic checks for values that can be validated reliably in
Python. Use LLM interpretation only where natural-language meaning
genuinely requires it; Python applies the resulting verification rules.

### Risk flags

Continue monitoring the defined compliance concerns, including: -
Fixed-deposit confusion (`FD_CONFUSION`). - Guarantee claims
(`GUARANTEE_CLAIM`). - Signing pressure (`SIGN_PRESSURE`). - Third-party
influence (`THIRD_PARTY`).

Each flag should include the source transcript, timestamp, rule or
reason, and session ID.

### Verdict authority

Python alone calculates the final Green / Amber / Red result according
to the project's approved rules.

-   **Green:** All mandatory checks meet the defined first-attempt pass
    requirements, with no disqualifying flags or unresolved uncertainty.
-   **Amber:** The defined review conditions apply, such as retries or
    unresolved uncertainty.
-   **Red:** The defined failure, refusal or critical risk conditions
    apply.

The exact mapping must follow the existing project plan and compliance
rules. Do not invent a new verdict policy in the LLM prompt.

Every verdict must include machine-readable reasons and supporting
evidence. Never silently convert an unclear answer into a pass.

------------------------------------------------------------------------

## 7. Synthetic policy knowledge

For the demo, use one synthetic insurance product with a complete and
internally consistent policy record.

Store the approved facts in a structured JSON file or similarly simple
configuration. Include only facts needed for the demo and verification
checks, such as: - Product name and category. - Premium amount and
frequency. - Premium payment term. - Policy term. - Benefits and payout
details. - Payout recipient rules. - Discontinuance / stopping-premium
conditions. - Relevant exclusions and limitations. - Approved
plain-language explanations.

`get_policy_fact` should return only information supported by this data.

If a requested fact is missing, the agent must say it cannot confirm the
information and offer escalation rather than invent an answer.

A vector database or elaborate RAG system is not required for one
synthetic policy.

------------------------------------------------------------------------

## 8. Demo scenario: 40--60 seconds

Build and rehearse one compelling end-to-end scenario. The recommended
story demonstrates conversational intelligence, tool use and compliance.

  -----------------------------------------------------------------------
  Approx. time            Demo moment             What it demonstrates
  ----------------------- ----------------------- -----------------------
  0--8 sec                Greeting, purpose and   Professional voice
                          consent                 persona

  8--20 sec               Confirm one policy fact Verification and
                                                  evidence recording

  20--33 sec              Customer interrupts:    Intent understanding,
                          "Ye FD hai kya?"        policy lookup, natural
                                                  explanation and
                                                  pending-question memory

  33--43 sec              Customer makes a        Risk recognition and
                          guarantee-related       genuine `flag_risk`
                          statement               tool call

  43--55 sec              Python evaluates        Independent compliance
                          evidence and dashboard  decision and traceable
                          shows outcome           verdict
  -----------------------------------------------------------------------

Important: - The final verdict must follow the actual configured rules.
A guarantee-related statement alone must not be forced to produce Red
unless the approved rules say so. - Use synthetic customer and policy
data only. - Do not hardcode the dashboard result or fabricate tool
activity. - The agent must actually execute the tools shown in the
demo. - Keep the scenario short and rehearsed, but do not conceal system
failures or misrepresent the result.

------------------------------------------------------------------------

## 9. API and dashboard workstream --- separate teammate ownership

The dashboard/API work should proceed in parallel with the voice-agent
redesign, using an agreed API contract. The teammate should not need to
modify the voice agent's internal conversation logic.

### Responsibility split

  Voice-agent owner                    API/dashboard teammate
  ------------------------------------ ---------------------------------------
  LiveKit session and audio pipeline   Backend API and persistence
  STT, LLM and TTS integration         Dashboard UI
  Agent prompts and tool calling       Session and event endpoints
  Conversation context                 Live transcript/event display
  Verification and compliance logic    Verification status display
  Verdict calculation                  Verdict and evidence display
  Emit structured events               Consume and present structured events
  Voice-agent tests                    API, dashboard and integration tests

Agree on the contract before either side finalises its implementation.

### Suggested backend responsibilities

The backend should: - Create and retrieve demo sessions. - Store session
metadata and synthetic customer references. - Receive structured
conversation and tool events. - Store transcript segments and
timestamps. - Store verification results and risk flags. - Store final
verdict and supporting reasons. - Provide read APIs for the dashboard. -
Support session status and error reporting. - Keep dashboard/API code
independent from the LiveKit worker internals.

Use authentication or scoped demo-session tokens for write endpoints.
Validate session IDs and payloads. Do not expose secrets or unrestricted
write access to the browser.

### Suggested API contract

These are proposed interface examples; the teammate should confirm
endpoint names and payloads with the voice-agent owner before
implementation.

  -------------------------------------------------------------------------------------------
  Method                  Endpoint                                    Purpose
  ----------------------- ------------------------------------------- -----------------------
  `POST`                  `/api/sessions`                             Create a demo session

  `GET`                   `/api/sessions/{session_id}`                Retrieve session
                                                                      details

  `POST`                  `/api/sessions/{session_id}/events`         Ingest structured agent
                                                                      events

  `GET`                   `/api/sessions/{session_id}/events`         Retrieve event timeline

  `GET`                   `/api/sessions/{session_id}/transcript`     Retrieve transcript

  `GET`                   `/api/sessions/{session_id}/verification`   Retrieve check statuses
                                                                      and evidence

  `GET`                   `/api/sessions/{session_id}/verdict`        Retrieve final verdict
                                                                      and reasons

  `POST`                  `/api/sessions/{session_id}/complete`       Mark session complete
  -------------------------------------------------------------------------------------------

For a short demo, polling a read endpoint is acceptable if it is
reliable. Server-Sent Events or WebSockets can be added if the dashboard
needs genuinely live updates and the implementation time permits. Do not
build complex real-time infrastructure unnecessarily.

### Event contract

The voice agent should emit structured events. Suggested event types:

-   `session.started`
-   `transcript.final`
-   `agent.response`
-   `intent.detected`
-   `tool.started`
-   `tool.completed`
-   `answer.recorded`
-   `verification.updated`
-   `risk.flagged`
-   `handoff.requested`
-   `verdict.calculated`
-   `session.completed`
-   `session.error`

Example event:

``` json
{
  "event_id": "evt_001",
  "session_id": "demo_001",
  "event_type": "risk.flagged",
  "timestamp": "2026-01-01T10:00:00Z",
  "data": {
    "flag": "GUARANTEE_CLAIM",
    "evidence": "Mujhe toh bola tha ki returns guaranteed hain.",
    "source": "customer_transcript"
  }
}
```

The timestamp and values above are illustrative only.

Events should be: - Session-scoped. - Timestamped. - Structured and
versionable. - Idempotent or deduplicated using `event_id`. - Grounded
in actual agent/tool activity.

### Dashboard requirements

Prioritise a clean demo dashboard, not a large analytics platform.

**Live session panel** - Session status. - Synthetic customer/policy
identifier. - Live transcript with speaker labels. - Agent response and
current stage.

**Agent activity panel** - Detected intent. - Actual tool calls and
results. - Policy fact retrieval. - Pending question. - Risk flags.

**Verification panel** - Mandatory checks and statuses. - Customer
answer. - Normalised answer. - Supporting transcript evidence. - Retry
count or review reason.

**Final outcome panel** - Green / Amber / Red. - Explicit reasons. -
Supporting evidence. - Human-review status where applicable.

The dashboard must display actual backend data. It must not
independently calculate or override the verdict.

### Integration agreement

Before implementation, both teammates should agree on: 1. Session ID
format and ownership. 2. Event names and payload schemas. 3. API
authentication method. 4. Transcript format and timestamps. 5.
Verification status vocabulary. 6. Verdict schema and reason format. 7.
Error and retry behaviour. 8. API versioning and compatibility.

The voice agent should be able to continue its local verification safely
if the dashboard/API temporarily becomes unavailable. Dashboard
persistence must not block the real-time audio response path.

------------------------------------------------------------------------

## 10. Latency plan

Optimise end-to-end response time, not just LLM generation time.

Measure these timestamps independently: 1. Customer speech end /
turn-finalised time. 2. STT final transcript available. 3. LLM request
start. 4. LLM first token. 5. Tool start and completion. 6. Final
response text ready. 7. TTS request start. 8. First audio chunk
received. 9. First audio played to customer.

### Initial performance goals

These are targets for testing, not guaranteed provider timings.

  -----------------------------------------------------------------------
  Metric                                                   Initial target
  ------------------------------ ----------------------------------------
  Completed speech to first                   Stretch goal: ≤ 1.2 seconds
  audible response               

  Local Python tool execution                            Ideally \< 50 ms

  LLM time to first token              Measure; aim for 300--600 ms where
                                                                 feasible

  Unnecessary model calls                                            Zero

  Unnecessary                                                        Zero
  translation/transliteration    
  calls                          

  Long dead air                                                     Avoid
  -----------------------------------------------------------------------

### Latency tactics

-   Use streaming STT and TTS where supported by the actual integration.
-   Use correct turn detection; avoid excessive silence before
    processing.
-   Keep system prompts concise and avoid sending the entire
    conversation repeatedly if a compact context is sufficient.
-   Keep policy facts structured and locally available.
-   Keep local tools fast and avoid external API calls for simple
    checks.
-   Generate concise spoken responses.
-   Begin TTS as soon as a safe, complete phrase is available, if
    supported.
-   Cache only fixed greetings and approved static fallback phrases.
-   Do not cache or fake dynamic verification responses.
-   Measure before optimising; do not change providers based on guesses.

The Devanagari output policy should not require a second LLM call. It
may slightly affect generation characteristics, but should not introduce
a separate translation stage.

------------------------------------------------------------------------

## 11. Testing and acceptance criteria

### Conversation tests

Test at least: - Normal answer. - Customer asks a policy question
mid-answer. - Customer misunderstands. - Customer corrects a previous
answer. - Customer interrupts the agent. - Multiple answers in one
utterance. - Unclear or incomplete response. - Silence. - Refusal. -
Unsupported policy question. - Guarantee-related statement. -
Third-party pressure. - Customer asks the same question repeatedly.

### Language tests

Test: - Hindi in Devanagari. - Hinglish code-mix. - English insurance
terms within Hindi sentences. - Amounts, dates and durations. - Names
and common Hindi pronunciation edge cases. - Same-sentence TTS
comparison across script formats.

### Compliance tests

Verify: - Consent refusal is respected. - Mandatory checks cannot be
skipped. - Unclear answers do not become passes. - Retry limits are
enforced. - Risk flags retain transcript evidence. - LLM cannot override
verification results. - Verdict is computed only by Python. - Dashboard
shows the same verdict and evidence as the backend.

### Demo acceptance checklist

-   [ ] Agent sounds natural in Hindi/Hinglish.
-   [ ] Hindi words are generated in Devanagari.
-   [ ] Customer questions do not break the verification flow.
-   [ ] Pending questions are preserved and resumed.
-   [ ] Real tool calls execute successfully.
-   [ ] Risk flags contain supporting evidence.
-   [ ] Python calculates the correct verdict.
-   [ ] No mandatory checks are silently skipped.
-   [ ] Dashboard displays actual session events.
-   [ ] API/dashboard failure does not break the voice response path.
-   [ ] End-to-end latency is measured.
-   [ ] The complete demonstration fits within 40--60 seconds.
-   [ ] No fabricated activity or verdict is shown.

------------------------------------------------------------------------

## 12. Implementation roadmap

Work in this order:

### Phase 1 --- Read-only audit

Inspect the current repository, identify working integrations and actual
conversation failures. No code changes.

### Phase 2 --- Agree on API contract

Voice-agent owner and dashboard teammate agree on session IDs, event
schema, API payloads and verdict representation.

### Phase 3 --- Conversational agent

Implement the single LLM agent, system instructions, session context,
tool registry and pending-question preservation.

### Phase 4 --- Tools and compliance

Connect the tools to real Python operations. Preserve mandatory checks,
risk rules, evidence and verdict authority.

### Phase 5 --- Policy knowledge and language

Add the synthetic policy facts and Hindi/Hinglish script policy. Test
TTS pronunciation before making provider changes.

### Phase 6 --- API and dashboard integration

Build the session/event APIs and live dashboard in parallel. Integrate
using the agreed contract.

### Phase 7 --- Latency and reliability

Instrument the complete pipeline and fix the largest measured
bottlenecks.

### Phase 8 --- Demo rehearsal

Run repeated end-to-end calls, verify the actual outcome and capture a
clean 40--60 second demo.

------------------------------------------------------------------------

## 13. Explicitly out of scope for this demo

Do not spend time on: - Multiple collaborating LLM agents. - Vector
databases and elaborate RAG. - Support for many insurance products. -
Large-scale analytics and reporting. - Complex workflow platforms. -
Unnecessary microservices. - Provider migrations without evidence. -
Separate translation LLMs. - Fake tool traces or hardcoded demo
verdicts. - Broad infrastructure work unrelated to the demonstration.

Use synthetic data only. Do not use real customer policies, personal
information or recordings. Do not claim regulatory certification or
production compliance based solely on this demo.

------------------------------------------------------------------------

## 14. Team ownership summary

  Workstream                            Primary owner
  ------------------------------------- ------------------------
  Voice agent, LLM, prompts and tools   Voice-agent developer
  Python verification and verdict       Voice-agent developer
  STT/TTS language pipeline             Voice-agent developer
  Backend APIs and persistence          Dashboard/API teammate
  Dashboard UI and live event display   Dashboard/API teammate
  API contract and integration tests    Shared
  End-to-end demo and rehearsal         Shared

Both workstreams should integrate against the same agreed API/event
contract.

------------------------------------------------------------------------

## 15. Final scope decision

The project will be built around:

**One intelligent conversational insurance agent + real tool calling +
Python-controlled compliance + a separate API/dashboard workstream.**

The demo will focus on one synthetic policy and one carefully designed
40--60 second scenario.

The success criteria are not the number of features or agents. They
are: - The conversation feels intelligent and natural. - The voice is
clear and fast. - The agent uses tools for real actions. - The
verification process remains trustworthy. - The dashboard makes the
agent's intelligence and evidence visible.

**Build the smallest genuine agent that can deliver the strongest honest
demonstration.**
