# Train on Vast.ai

The scripts use on-demand SSH instances for finite training runs. The launcher uses an **existing instance ID**, uploads a source allowlist, supervises setup and training, verifies downloaded artifacts, and requests compute stop on completion or failure. It never rents, replaces, or destroys an instance.

## Prepare locally

From the repository root, use the Python environment that already loads the experiment presets:

```bash
uv tool install vastai
engine/.venv/bin/python scripts/vastai/cloud_train.py --help
```

Set `VAST_API_KEY` privately in the process environment, or use the official CLI's saved authentication. Do not put a key in commands, YAML, source bundles, or chat. Check authentication with `vastai show user --raw` locally; that output contains account details and should not be committed.

Ensure the account has the public key matching your local SSH private key. Inspect `vastai show ssh-keys --raw`. If registration is needed, `vastai create ssh-key` takes public-key **contents**, not a filename. Use a subprocess argument array and verify the resulting key list; never upload a private key. Unlock a passphrase-protected identity with `ssh-add` before starting.

## Search and rent explicitly

```bash
engine/.venv/bin/python scripts/vastai/cloud_train.py offers --storage 80 --limit 20
vastai show instances --raw
```

Search queries are configurable with `--query`. Compare actual allocated CPU, RAM, VRAM, bandwidth, disk performance, reliability, driver support and rental duration. Raw memory fields may use MB although query filters use GB. Search uses storage-inclusive `dph_total` for the requested disk allocation. Prices and availability are queried live; no GPU or hourly price is assumed.

Choose finite training steps, disk size, running duration and spending limits before creating capacity. Check for an existing suitable instance first. After selecting an offer and registering the key, rent explicitly with the official CLI:

```bash
vastai create instance OFFER_ID --image vastai/pytorch:@vastai-automatic-tag \
  --disk 80 --ssh --direct --cancel-unavail --label aresim-train --raw
```

Record `new_contract` as the instance ID immediately. Do not publish the complete response: it can contain an instance API key. Inspect stdout and stderr even if the CLI exit code is zero. If creation has an uncertain outcome, inspect existing instances before any retry. Wait for the instance to report running and expose direct SSH. Rental/startup time before the training launcher is additional billed time.

## Inspect the upload and run

This example uses the checked-in DQN smoke workload with a GPU learner. The dollar figures are user-selected ceilings, not advertised marketplace prices.

```bash
engine/.venv/bin/python scripts/vastai/cloud_train.py bundle \
  --config configs/masked_dqn/smoke.yaml \
  --set resources.gpus_per_learner=1 \
  --output /tmp/aresim-vast-payload

engine/.venv/bin/python scripts/vastai/cloud_train.py run --on INSTANCE_ID \
  --config configs/masked_dqn/smoke.yaml \
  --set resources.gpus_per_learner=1 \
  --hours 0.25 --max-hourly 0.60 --max-cost 0.25
```

`bundle` writes a new directory for review without contacting Vast.ai. Engine Python source, `pyproject.toml`, YAML configs, seed manifests and required setup/supervision scripts are included. Environment files, keys, results, caches, notebooks and the web application are excluded. A SHA-256 source inventory records exactly what was uploaded. Existing staging directories are rejected.

For PPO or recurrent DQN, set `resources.gpus_per_learner=1`. For MuZero, DreamerV3 or either Plan2Explore phase, set `resources.device=cuda`. Use reference configurations only after measuring the smoke workload on the selected machine, and tune collector counts to the allocated CPU/RAM. Training checks NVIDIA visibility and executes a real CUDA operation before starting the selected `aresim-rl train` command.

The launcher validates the typed configuration and real seed manifest before upload. It starts fresh trials under a unique remote directory. Automatic checkpoint resume is not performed: that requires the original source and complete trainer artifacts.

For Plan2Explore adaptation, supply an explicit local pretrained export with `--init-from /absolute/path/to/checkpoint.json` to either `bundle` or `run`. The launcher validates the phase, configuration, frozen environment/seed compatibility and checksums, then uploads only the initializer sidecar and its `native/model.pt` / `native/trainer.pt`. It rewrites the initialization path inside the bundle. The local Python environment needs the world-model extra to validate this input. For example:

```bash
engine/.venv/bin/python scripts/vastai/cloud_train.py bundle \
  --config configs/plan2explore/adapt/smoke.yaml \
  --set resources.device=cuda \
  --init-from /absolute/path/to/pretrain/checkpoints/final/checkpoint.json \
  --output /tmp/aresim-vast-adapt
```

Use the same `--init-from` with `run` and its required instance/time/cost options. A smoke parent must match the smoke model; a reference parent must match the reference model. Each phase has its own finite interaction budget; account for both phases when comparing total training cost.

Tracking defaults remain in YAML. Select `--set tracking.mode=offline` or `--set tracking.mode=disabled` for this launcher; it does not transfer tracking credentials. The launcher does not upload the account-wide Vast key. The supervisor reads only `CONTAINER_ID` and the matching `CONTAINER_API_KEY` from startup environment when needed, then uses that scoped key for self-stop.

SSH accepts a new host key into the normal known-hosts store and rejects changes to an existing key. Use `--identity /absolute/path/to/private-key` if needed; the file stays local. Check an unfamiliar fingerprint through the provider console before first use if your SSH configuration requires interactive verification.

## Deadlines and recovery

`--hours` bounds upload and remote setup/training, including dependency installation. Local bundle preparation finishes before this deadline begins; inspect a bundle before renting when possible. `--grace-seconds` (default 30) allows workers to terminate before escalation. `--recovery-seconds` (default 300) bounds the recovery window before self-stop. The running-cost check includes these windows and shutdown allowance. Subsequent storage, transfer, and pre-launch rental costs are separate.

The supervisor starts detached before package setup. Both supervisor and client enforce deadlines and request instance stop. If setup fails before the remote CLI is installed, the supervisor uses the documented instance-management API with its scoped key for self-stop. Keep the local launcher connected until completion. Ctrl+C requests compute stop; outputs remain on the instance disk. A lost client connection still leaves the remote supervisor's deadline active, provided its scoped credentials and provider access work. Provider failures can prevent self-stop; inspect the local lifecycle record and stop the exact instance manually if needed.

Local records live under `results/cloud/vastai/INSTANCE_TIMESTAMP/lifecycle.json`. They retain public instance details, source identity, configuration, status, recovery checksums and the stop outcome. Downloaded results live under the sibling `artifacts/` directory. Completion is reported only after inspecting remote status and verifying every downloaded file.

If transfer fails or the instance stops before recovery completes, preserve its storage. Start that same instance explicitly, recover using the saved lifecycle record, then stop it:

```bash
vastai start instance INSTANCE_ID --raw
engine/.venv/bin/python scripts/vastai/cloud_train.py fetch \
  results/cloud/vastai/INSTANCE_TIMESTAMP/lifecycle.json
vastai stop instance INSTANCE_ID --raw
```

Stopping retains data and continues storage billing. After verified recovery, destroy storage only when you intentionally authorize discarding the remote copy, using the official CLI. The scripts never destroy instances or unrelated volumes.

Implementation: [`cloud_train.py`](../../scripts/vastai/cloud_train.py), [`supervisor.py`](../../scripts/vastai/supervisor.py), shared [`cloud_payload.py`](../../scripts/cloud_payload.py), and [`remote_setup.sh`](../../scripts/cloud/remote_setup.sh). JarvisLabs reuses the payload/setup owners through its existing entrypoints.

Provider references: [documentation index](https://docs.vast.ai/llms.txt), [authentication](https://docs.vast.ai/cli/authentication), [offer search](https://docs.vast.ai/cli/reference/search-offers), [instance creation](https://docs.vast.ai/cli/reference/create-instance), [container credentials and self-management](https://docs.vast.ai/guides/instances/docker-environment), [instance-management API](https://docs.vast.ai/api-reference/instances/manage-instance), and [stopping instances](https://docs.vast.ai/cli/reference/stop-instance).
