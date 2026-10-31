# Agent status

`core.py` contains the deterministic call stages, evaluation, retries, flag detection, and verdict rules. `workflow.py` drives those rules using the `VoiceIO` protocol and fixed prompts. The workflow is provider-independent and has mocked tests.

`LiveKitGnaniConfig` remains lazy so unit tests do not import the provider. `livekit_worker.py` now exposes a concrete `AgentServer` entrypoint and `LiveKitAgentSessionPort`. The local `.venv` package files are present, but the recorded Python 3.11.9 base interpreter is missing and Windows denied source reads for the installed packages; the worker APIs were checked against official LiveKit references, but still require a local startup check before a call.

The workflow's `listen()` contract returns one committed final string. LiveKit exposes final/interim status through `user_input_transcribed`, but the verified event has no committed-turn identifier or late marker. Late-transcript logging, browser publication with `late: true`, and affected-check re-evaluation/re-ask therefore remain unimplemented until turn correlation is verified. The concrete port uses LiveKit's normal interruption path and explicit `interrupt(force=True)` for cancellation.

`runtime_support.py` provides provider-independent helpers for stable synthesis cache keys, listener-approved in-memory audio entries, and monotonic turn timing. The timer returns missing STT timing as `None` when no final transcript mark was supplied; these helpers are not connected to a provider session or persistent metrics store.

`livekit_worker.py` builds the configured Gnani providers, starts the room-bound session with `AgentSession.start(..., room=ctx.room)`, runs the conversational workflow, uses `session.say` for TTS, publishes JSON events with `room.local_participant.publish_data`, and closes the session in a `finally` block. The legacy deterministic `CallWorkflow` entry point remains available for compatibility. `synthetic_call_data()` is only the local development default and must be replaced by the agreed synthetic scenario loader before a manual call.

Next verification should start the worker from the repaired Python 3.11.9 environment and exercise one synthetic room. Keep `VerificationEngine` authoritative for answer evaluation and final verdicts; `core.py` remains the source of check definitions and normalization rules.

## GO 6 manual startup

After restoring the Python 3.11.9 interpreter recorded by `.venv`, start the
worker with:

```powershell
.\.venv\Scripts\python.exe -m agent.livekit_worker dev
```

Use `start` instead of `dev` for the normal LiveKit worker mode. The default
entrypoint loads the synthetic Vansh/Suraksha Pension Plus scenario. A
customer must join the configured LiveKit room through the existing browser or
room client.

Required environment variable names are `LIVEKIT_URL`, `LIVEKIT_API_KEY`,
`LIVEKIT_API_SECRET`, and `GNANI_API_KEY`. LiveKit Inference uses the LiveKit
credentials; no `GEMINI_API_KEY` is required. `GEMINI_MODEL` may override the
default `google/gemini-3.5-flash-lite` model. Gnani Prisma `hi-IN` streaming
STT and Timbre `v2.5` `hi-IN` WebSocket TTS are configured in code.

The local `.venv` is currently not runnable because its recorded base Python
installation is missing. No startup or provider verification has been run.
