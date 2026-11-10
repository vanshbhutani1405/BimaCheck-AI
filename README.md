<div align="center">

# 🛡️ BimaCheck AI

### Voice-based insurance policy verification where **Python decides** and **Gemini only listens**

<br>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LiveKit](https://img.shields.io/badge/LiveKit-Agents_1.8.10-0B0B0B?style=for-the-badge&logo=livekit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_3.5-Flash--Lite-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white)
![Gnani](https://img.shields.io/badge/Gnani-Prisma_STT_·_Timbre_TTS-7C3AED?style=for-the-badge)
![Tests](https://img.shields.io/badge/tests-153_passed-2EA44F?style=for-the-badge&logo=pytest&logoColor=white)
![Language](https://img.shields.io/badge/Hindi_·_Hinglish-hi--IN-FF9933?style=for-the-badge)

<br>

<img src="architecture.png" alt="BimaCheck AI architecture" width="100%">

</div>

<br>

## ⚡ The Pipeline

```mermaid
flowchart LR
    A([🎙️ Customer speech]) --> B[LiveKit<br/>AgentSession]
    B --> C[Gnani Prisma<br/>STT · hi-IN]
    C --> D{{Gemini<br/>language understanding}}
    D --> E[🐍 Python<br/>checks · state · verdict]
    E --> F[Gnani Timbre<br/>TTS · timbre-v2.5]
    F --> G([🔊 LiveKit audio])

    style E fill:#16a34a,color:#fff,stroke:#14532d,stroke-width:3px
    style D fill:#4285F4,color:#fff,stroke:#1e3a8a
    style C fill:#7C3AED,color:#fff,stroke:#4c1d95
    style F fill:#7C3AED,color:#fff,stroke:#4c1d95
    style B fill:#111827,color:#fff,stroke:#000
```

> **Gemini interprets and proposes wording. Python alone owns the current question, state, evidence, policy comparisons, discrepancies, safety decisions, and the final verdict.** Gemini cannot pass a check or advance the workflow.

<br>

<table>
<tr>
<td width="33%" valign="top">

### 🧠 Gemini does
- Understands Hindi and Hinglish
- Proposes spoken wording
- Suggests an interpretation

**Never** verifies, passes, or advances.

</td>
<td width="33%" valign="top">

### 🐍 Python does
- Owns check ID and state
- Compares against the policy record
- Records discrepancies
- Runs safety rules
- Issues the verdict

**Always** uses the original transcript.

</td>
<td width="33%" valign="top">

### 🔒 Guardrails
- Name withheld from Gemini until verified
- Mandatory answers beat handoffs
- End-call needs explicit customer words
- Parse or provider failure gives a safe clarification, no retry

</td>
</tr>
</table>

<br>

## 🔍 Seven Checks, Four Turns

```mermaid
flowchart LR
    S([Consent]) --> I[Name match] --> P[Policy facts<br/>disclosed + confirmed]
    P --> T1["<b>Turn 1</b><br/>① Annual premium<br/>② Payment period"]
    T1 --> T2["<b>Turn 2</b><br/>③ Policy term<br/>④ Product type"]
    T2 --> T3["<b>Turn 3</b><br/>⑤ Payout recipient<br/>⑥ Pension timing"]
    T3 --> T4["<b>Turn 4</b><br/>⑦ Stopping premiums"]
    T4 --> V{{Verdict}}

    style V fill:#16a34a,color:#fff,stroke:#14532d,stroke-width:3px
    style T1 fill:#eff6ff,stroke:#3b82f6,color:#111
    style T2 fill:#eff6ff,stroke:#3b82f6,color:#111
    style T3 fill:#eff6ff,stroke:#3b82f6,color:#111
    style T4 fill:#eff6ff,stroke:#3b82f6,color:#111
```

Each check is evaluated independently. A correct answer passes and advances. A wrong, unclear, or refused answer becomes a **discrepancy** and advances, with no second substantive attempt. Compatible fragments merge with a **0.4s** max wait, unrelated turns are buffered, and end-call language is never merged.

<br>

## 🚦 Verdicts

<table>
<tr>
<td align="center" width="33%">

### 🟢 Green
`clear`

All checks passed, no review concerns.

</td>
<td align="center" width="33%">

### 🟡 Amber
`pending_review`

Discrepancies or non-critical review conditions.

</td>
<td align="center" width="33%">

### 🔴 Red
`hold`

Refused consent, or confirmed guarantee or sign-pressure risk.

</td>
</tr>
</table>

<br>

<table>
<tr>
<td width="50%" valign="top">

## 📋 Demo Policy Record

| Fact | Value |
|:--|:--|
| **Product** | Suraksha Pension Plus (deferred annuity) |
| **Annual premium** | ₹1,50,000 |
| **Payment period** | 10 years |
| **Policy term** | 20 years |
| **Payout recipient** | Sunita |
| **Payout starts** | After 5 years |
| **If premiums stop** | Policy may lapse or benefits may reduce |

<sub>The 5-year payout start is separate from the 10-year payment period.</sub>

</td>
<td width="50%" valign="top">

## 🇮🇳 Built for Indian Speech

| Input | Understood as |
|:--|:--|
| `₹1,50,000` | 150000 |
| `150000` | 150000 |
| `डेढ़ लाख` | 150000 |
| `Sunita` / `सुनीता` | Recipient match |
| `Vansh` / `वंश` | Name match |

<sub>Indian comma grouping, currency prefixes, and Hindi/Hinglish number words are normalized. Name matching is case-insensitive substring, not fuzzy phonetics.</sub>

</td>
</tr>
</table>

<br>

<table>
<tr>
<td width="50%" valign="top">

## 🧰 Stack

| Layer | Tech |
|:--|:--|
| Runtime | Python 3.11 (3.11.9) |
| Orchestration | `livekit-agents==1.8.10` |
| Speech | `livekit-plugins-gnani==1.8.10` |
| STT | Gnani Prisma streaming, `hi-IN` |
| TTS | Gnani Timbre `timbre-v2.5`, WebSocket |
| LLM | `google/gemini-3.5-flash-lite` via LiveKit Inference |
| LLM config | Low reasoning, 8s timeout, no retry |
| Policy | JSON record + deterministic evaluators |
| Tests | pytest |

</td>
<td width="50%" valign="top">

## 📈 Observability and Tests

Monotonic logs across every stage:

`Final STT` → `Gemini request/response` → `Python evaluation` → `TTS request/start` → `Playback done or interrupted`

<br>

| Suite | Result |
|:--|:--|
| Full | ![](https://img.shields.io/badge/153-passed-2EA44F) |
| Voice, policy, compliance | ![](https://img.shields.io/badge/80-passed-2EA44F) |

<sub>`TOTAL_TO_FIRST_AUDIO_MS` reports `unavailable`, since first-audio onset is not reliably measurable.</sub>

</td>
</tr>
</table>

<br>

<details>
<summary><b>🔬 Gemini boundary, in detail</b></summary>

<br>

- The prompt and schema describe **ten fields**. The parser requires a non-empty `spoken_response` and a valid `intent`, defaults the rest, ignores extra top-level fields, and rejects invalid enums, types, or tool shapes.
- Python pins the check ID and the original transcript. Gemini's candidate is interpretation only, never verification evidence.
- A proposed handoff cannot override a current mandatory answer.
- Risk signals are checked by Python against approved patterns.

</details>

<details>
<summary><b>🛠️ Key implementation highlights</b></summary>

<br>

Deterministic identity routing · Python-owned check state and verdicts · suppression of Gemini-directed transitions · corrected handoff precedence · Hindi/Indian-number and recipient matching · four-turn aggregation with bounded waiting · one-pass discrepancy handling · tolerant Gemini response parsing · turn-level latency instrumentation

</details>

<details>
<summary><b>🧪 Run the tests</b></summary>

<br>

```bash
pip install livekit-agents==1.8.10 livekit-plugins-gnani==1.8.10 pytest
pytest
```

</details>

<br>

<div align="center">

**Scope:** this repo contains the voice agent only. No database, backend, or frontend.

<sub>Built by Vansh</sub>

</div>