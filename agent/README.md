# Agent status

`core.py` contains the deterministic call stages, evaluation, retries, flag detection, and verdict rules. `workflow.py` drives those rules using the `VoiceIO` protocol and fixed prompts. The workflow is provider-independent and has mocked tests.

`LiveKitGnaniConfig` remains lazy so unit tests do not import the provider. `livekit_worker.py` now exposes a concrete `AgentServer` entrypoint and `LiveKitAgentSessionPort`. The local `.venv` package files are present, but the recorded Python 3.11.9 base interpreter is missing and Windows denied source reads for the installed packages; the worker APIs were checked against official LiveKit references, but still require a local startup check before a call.

The workflow's `listen()` contract returns one committed final string. LiveKit exposes final/interim status through `user_input_transcribed`, but the verified event has no committed-turn identifier or late marker. Late-transcript logging, browser publication with `late: true`, and affected-check re-evaluation/re-ask therefore remain unimplemented until turn correlation is verified. The concrete port uses LiveKit's normal interruption path and explicit `interrupt(force=True)` for cancellation.

`runtime_support.py` provides provider-independent helpers for stable synthesis cache keys, listener-approved in-memory audio entries, and monotonic turn timing. The timer returns missing STT timing as `None` when no final transcript mark was supplied; these helpers are not connected to a provider session or persistent metrics store.

`livekit_worker.py` builds the configured Gnani providers, starts the room-bound session with `AgentSession.start(..., room=ctx.room)`, runs `CallWorkflow`, uses `session.say` for TTS, publishes JSON events with `room.local_participant.publish_data`, and closes the session in a `finally` block. `synthetic_call_data()` is only the local development default and must be replaced by the agreed synthetic scenario loader before a manual call.

Next verification should start the worker from the repaired Python 3.11.9 environment and exercise one synthetic room. Keep `core.py` authoritative for all call decisions and follow the plan for event contracts and late transcripts.
