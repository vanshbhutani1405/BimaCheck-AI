# Agent status

`core.py` contains the deterministic call stages, evaluation, retries, flag detection, and verdict rules. `workflow.py` drives those rules using the `VoiceIO` protocol and fixed prompts. The workflow is provider-independent and has mocked tests.

`LiveKitGnaniConfig` and `LiveKitIO` are currently configuration/adapter seams. There is no verified LiveKit worker entrypoint, `AgentSession` lifecycle, microphone/audio-track binding, or browser data-channel binding yet. The local `.venv` package files are present, but the recorded Python 3.11.9 base interpreter is missing and Windows denied source reads for the installed packages. Consequently, exact SDK signatures could not be independently checked in this workspace; no concrete session calls are claimed or added.

The workflow's `listen()` contract returns one string. It does not expose partial/final transcript events, turn identifiers, or a callback for transcripts arriving after a turn is committed. Late-transcript logging, browser publication with `late: true`, and affected-check re-evaluation/re-ask are therefore not implemented. The current `stop_playback()` method is only a transport contract; actual LiveKit barge-in remains unbound.

`runtime_support.py` provides provider-independent helpers for stable synthesis cache keys, listener-approved in-memory audio entries, and monotonic turn timing. The timer returns missing STT timing as `None` when no final transcript mark was supplied; these helpers are not connected to a provider session or persistent metrics store.

`livekit_worker.py` now provides the concrete lifecycle around a small `SessionPort`: it builds the configured Gnani providers, starts the session, runs `CallWorkflow`, cancels active playback before replacement, publishes workflow events through the port, and closes the session in a `finally` block. It does not claim to bind LiveKit room tracks or SDK callbacks; that binding remains the deployment-specific implementation of `SessionPort`.

Next integration work should inspect the installed LiveKit 1.8.10 and Gnani 1.8.10 APIs from the project's working Python 3.11.9 installation, then bind a session adapter and test with mocks before any synthetic live call. Keep `core.py` authoritative for all call decisions and follow the plan for event contracts and late transcripts.
