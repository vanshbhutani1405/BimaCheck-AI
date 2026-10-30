# Evaluation baseline

This directory contains reproducible local tests and will hold measured evidence. No live provider call or benchmark is performed by the unit tests.

## Evidence status

- No audio fixtures, call logs, transcripts, credit reports, latency records, pronunciation results, or endpointing experiments were present in the repository.
- `test_runtime_support.py` tests deterministic synthesis-cache keying, the listener-approval guard for cached bytes, and timer arithmetic with a fake monotonic clock. It does not call or validate LiveKit/Gnani.
- The project `.venv` cannot currently start: its `pyvenv.cfg` points to `C:\Users\pc\AppData\Local\Programs\Python\Python311`, where the base interpreter is missing. The package source files are present but Windows denied reads. Do not replace this evidence with a test run from another Python environment.
- The installed `livekit-plugins-gnani==1.8.10` package is evidence of a prepared adapter, not evidence that a live request succeeded. Its documented interfaces are Prisma STT streaming (`wss://api.vachana.ai/stt/v3/stream`), Prisma STT REST (`/stt/v3`), and Timbre TTS REST/SSE/WebSocket (`/api/v1/tts`).
- The adapter supports `timbre-v2.5`, while its default is `timbre-v2.0`. The plan's required Timbre `v2.5` choice has not been exercised or pronunciation-checked in this repository.
- The first required live gate is the plan's Prisma stress test: noisy Hinglish numbers plus slow speech with mid-sentence pauses. It must record number accuracy, returned script, confidence behavior, credit use, and late transcripts.

The pronunciation set required by the plan belongs under `eval/pronunciation/` and must be generated with Timbre `v2.5`, listened to by a person, and marked before audio is used in the agent. No audio fixtures or approved cache entries exist yet.

## Go 3 provider gate

The current workflow remains transport-independent. There is no verified LiveKit worker/session/audio-track adapter. `VoiceIO.listen()` exposes a single returned answer, so the current interface cannot identify a transcript arriving after its turn is committed, publish it with `late: true`, or safely re-evaluate the affected check. Actual barge-in is also unverified: `stop_playback()` is only a mockable contract. Exact plugin/session method signatures remain unverified because the recorded Python 3.11.9 base interpreter is absent and installed package source reads are denied in this workspace.

Safe local verification command (run after the Python 3.11.9 base installation has been restored, without recreating `.venv`):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s eval -p "test_*.py" -v
```

## Required measurements still missing

STT latency, LLM time-to-first-token, TTS time-to-first-audio, end-to-end response latency, cold/warm split, event-loop lag, provider first-request versus later-request timing, endpointing cut-offs, late transcripts, confidence by noise level, credit use, number accuracy, and pronunciation verdicts. The timer unit tests validate only arithmetic; they are not measurements of these metrics.
