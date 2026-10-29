# BimaCheck AI

BimaCheck AI is a synthetic Hinglish voice-verification prototype. The authoritative scope and staged build rules are in [BimaCheck_AI_Codex_Plan.md](BimaCheck_AI_Codex_Plan.md).

## Current status

This repository is at Go 1: foundation and integration evidence capture. No agent, API, web app, database schema, conversation workflow, or live-call harness has been implemented yet.

The repository contains a local `.venv` created for the Gnani/LiveKit experiments. It must be preserved. On this machine its interpreter is not currently runnable because the Python 3.11 installation recorded in `.venv/pyvenv.cfg` is missing. Do not recreate or upgrade it as part of Go 1.

Observed package metadata in that environment includes:

- `livekit-agents==1.8.10`
- `livekit-plugins-gnani==1.8.10`
- `livekit==1.1.21`
- `livekit-api==1.2.2`
- `websockets==15.0.1`
- `python-dotenv==1.2.4`

These are observations, not a new lock file or a claim that the environment can currently execute them.

## Environment

PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

The local `.env` is intentionally untracked and ignored. It contains the existing LiveKit and Gnani credentials. Never print it, copy its values into logs, or commit it. The current key names observed without values are `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and `GNANI_API_KEY`.

## Integration status

The installed Gnani LiveKit adapter documents Prisma speech-to-text and Timbre text-to-speech through Gnani Vachana. Its supported interfaces are streaming STT over `wss://api.vachana.ai/stt/v3/stream`, REST STT at `/stt/v3`, and REST/SSE/WebSocket TTS at `/api/v1/tts`. It supports Timbre `v2.5`; the adapter's default TTS model is `timbre-v2.0`, so Go 2 must select `model="timbre-v2.5"` explicitly after the pronunciation baseline is prepared.

No repository evidence proves which transport, voice, exact language options, Gemini model, or live call was previously tested. The inspected environment did not contain a Gemini-specific SDK or a Gemini credential key; the claim that Gemini 3.5 Flash Lite was tested therefore remains unverified from this repository. No latency, pronunciation, confidence, credit, endpointing, or late-transcript measurement is recorded yet.

Official references:

- [Gnani LiveKit STT guide](https://docs.livekit.io/agents/models/stt/gnani/)
- [Gnani LiveKit Python reference](https://docs.livekit.io/reference/python/livekit/plugins/gnani/index.html)
- [Gnani API documentation](https://docs.gnani.ai/)

## Repository ownership

- `agent/` is reserved for Vansh's voice-agent implementation.
- `eval/` holds reproducible transcript/live-call evidence and pronunciation results.
- `db/` is a shared contract area owned with the teammate. Go 1 does not change `schema.sql` or `events.md`.

## Safe commands

Inspect the repository without exposing secrets:

```powershell
git status --short --branch
Get-ChildItem -Force
Get-Content .gitignore
Get-Content .env | ForEach-Object { if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') { $matches[1] } }
```

The Go 1 baseline has not been run because there is no harness or audio fixture and the local virtual-environment interpreter is unavailable. Do not make live Gnani calls until the environment is repaired and a short, credit-bounded test plan is agreed.

## Who built what

Go 1 establishes only the shared foundation and evidence notes. The future ownership split is documented in the plan: Vansh owns `agent/` and transcript tests; the teammate owns `api/`, `db/`, and `web/`.
