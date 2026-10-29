# Evaluation baseline

This directory is reserved for measured evidence. Go 1 records the baseline status only; it does not make live API calls or invent measurements.

## Evidence status

- No audio fixtures, call logs, transcripts, credit reports, latency records, pronunciation results, or endpointing experiments were present in the repository.
- No baseline harness was added because the local `.venv` interpreter is unavailable and there is no safe fixture to run against.
- The installed `livekit-plugins-gnani==1.8.10` package is evidence of a prepared adapter, not evidence that a live request succeeded. Its documented interfaces are Prisma STT streaming (`wss://api.vachana.ai/stt/v3/stream`), Prisma STT REST (`/stt/v3`), and Timbre TTS REST/SSE/WebSocket (`/api/v1/tts`).
- The adapter supports `timbre-v2.5`, while its default is `timbre-v2.0`. The plan's required Timbre `v2.5` choice has not been exercised or pronunciation-checked in this repository.
- The first required live gate is the plan's Prisma stress test: noisy Hinglish numbers plus slow speech with mid-sentence pauses. It must record number accuracy, returned script, confidence behavior, credit use, and late transcripts.

The pronunciation set required by the plan belongs under `eval/pronunciation/` and must be generated with Timbre `v2.5`, listened to by a person, and marked before audio is used in the agent.

## Required measurements still missing

STT latency, LLM time-to-first-token, TTS time-to-first-audio, end-to-end response latency, cold/warm split, event-loop lag, provider first-request versus later-request timing, endpointing cut-offs, late transcripts, confidence by noise level, credit use, number accuracy, and pronunciation verdicts.
