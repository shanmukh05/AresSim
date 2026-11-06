# Cloud execution ownership

Reviewed November 7, 2026. Existing working-tree changes are preserved. Optional cleanup includes comments, never unit tests.

- Share a source allowlist and remote dependency setup between GPU providers.
  - JarvisLabs staging copies whole engine/config directories, implicitly trusting that no local secrets or generated data were placed there.
  - A second GPU provider needs exactly the same install extras and runtime files; separate setup implementations would drift.
  - Seed manifests are runtime inputs but presets refer to an absent, ignored notebook directory. The training refactor moves recovered manifests into `configs/seeds/`.
- Keep Vast.ai account operations on the client and bounded work on the instance.
  - Provider credentials must never be part of a source bundle or remote training environment.
  - A trainer step budget alone cannot bound package setup, stalled workers, or compute billing.
  - A detached remote supervisor and a client deadline cover setup and training; results are recovered before stopping, with checksums. Only an explicitly selected existing instance is used. Rental remains an explicit official CLI operation.

The launcher records instance identity, source inventory, resolved configuration, work deadline, public price information, shutdown outcome and recovery status. It never destroys storage. Storage and transfer costs remain separate from the running hourly estimate.
## Cloud execution review


- Treat the supervisor status as mutable control data and checkpoint/log files as frozen recovery artifacts.
  - Hashing `cloud_status.json` before setting `artifacts_ready` guarantees a recovery checksum mismatch.
  - A failed training command still needs its artifact inventory recovered; an early `failed` state must distinguish command completion from supervisor startup failure.
- Apply one deadline to transfer, setup, training and artifact recovery; stop the selected instance in all execution exits.
  - Per-file transfer timeouts can exceed the declared aggregate recovery window.
  - Upload time must reduce the remote training deadline, and SIGTERM must reach the worker process group before self-stop.
- Keep provider credentials out of bundles, worker environments and subprocess output; validate remote inventory paths before materializing files.
  - The source allowlist excludes env files and symlinks. Recovery also needs to reject traversal, malformed hashes and pre-existing local symlink paths.

Cloud verification completed: all 18 non-adaptation training configurations stage correctly with GPU/offline overrides; the adaptation smoke bundle includes exactly the selected pretraining sidecar and its two verified native files. Incompatible development/reference parents are rejected before remote work. Fresh 64-step CPU pretraining and 64-step CPU adaptation completed using the shared transfer validator. Cloud preparation now freezes the resolved seed/schema contract before validating transfer, matching trainer behavior. Credential/result sentinels and source symlinks were excluded/rejected, shell metacharacters remained literal, a real subprocess timeout and process-group shutdown passed, and mutable status changes did not invalidate artifact hashes. CLI help, JarvisLabs staging self-check, Python compilation, shell syntax, and whitespace checks passed. No paid instance was created or started; live SSH transfer, CUDA setup, provider stop acknowledgments, and the REST self-stop fallback remain unexercised against a paid container. The fallback URL/authentication were checked against the official Vast instance-management documentation.
