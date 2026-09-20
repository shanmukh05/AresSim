# Train on JarvisLabs

Last updated: 2026-09-12

Run the same `aresim-rl train` path from [RL Usage](usage.md) on a JarvisLabs GPU container or CPU VM. The helper uploads a slim training payload, installs Python 3.12 plus `aresim[rllib]`, starts training, then downloads the entire remote `results/` tree. Official CLI reference: [JarvisLabs CLI](https://docs.jarvislabs.ai/cli/).

Do not use `jl run .` from the repository root. That syncs `results/`, `web/`, and local venvs, and JarvisLabs pytorch templates default to Python 3.10, which cannot install AresSim (`requires-python = ">=3.12"`).

## Contents

| Section | What you will find |
|---|---|
| [One-time setup](#one-time-setup) | `jl`, API token, SSH key in `ssh-add`, W&B key |
| [One-command training](#one-command-training) | `scripts/jarvislabs/cloud_train.py` |
| [Manual `jl` workflow](#manual-jl-workflow) | Create, upload, setup, train, download, pause |
| [GPU vs CPU](#gpu-vs-cpu) | Instance types and `--set` resource overrides |
| [What gets uploaded](#what-gets-uploaded) | Payload vs what stays local |
| [After the run](#after-the-run) | Fetch, inspect, resume, destroy |

---

## One-time setup

Do this **before** `cloud_train.py run`. `jl upload` uses scp even on GPU templates. A billed instance that cannot accept scp is still created, then paused on failure.

### 1. Install and authenticate `jl`

```bash
uv tool install jarvislabs   # or: pip install jarvislabs
jl setup                     # paste the token from jarvislabs.ai/settings/api-keys
jl status
jl gpus
```

`jl setup` stores the token in `~/Library/Application Support/jl/config.toml` on macOS. You can also export `JL_API_KEY`.

### 2. SSH key (required for GPU and CPU)

Two separate steps. Registering the key with JarvisLabs is not enough.

**A. Register once** — use a public key that actually exists (`ls ~/.ssh/*.pub`). This laptop has `id_rsa.pub`, not `id_ed25519.pub`:

```bash
ls ~/.ssh/*.pub
jl ssh-key add ~/.ssh/id_rsa.pub --name "my-laptop"
# if you have Ed25519 instead:
# jl ssh-key add ~/.ssh/id_ed25519.pub --name "my-laptop"
```

**B. Unlock in every new terminal** — passphrase-protected keys cannot be used by non-interactive scp. `ssh-add` only lasts for that session:

```bash
ssh-add ~/.ssh/id_rsa    # or ~/.ssh/id_ed25519; type the passphrase
ssh-add -l               # must list an rsa or ed25519 key. "The agent has no identities" will fail upload.
```

`cloud_train.py run` now refuses to create an instance if the agent has no unlocked identity. Skip `ssh-add` and you get `Permission denied (publickey)` after the GPU is already running.

### 3. W&B for online configs

Checked-in `dev.yaml` and `reference.yaml` use `tracking.mode: online`. The helper copies the key to `$HOME/aresim/.wandb.env` and does not print it. Either export it, or put `WANDB_API_KEY=...` in repo-root `.env.local` (the helper loads that file):

```bash
export WANDB_API_KEY=...     # from wandb.ai/authorize
```

### 4. Preflight in the same terminal

```bash
ssh-add -l                   # key listed
echo "$WANDB_API_KEY"        # non-empty for reference.yaml / dev.yaml
python3 scripts/jarvislabs/cloud_train.py self-check
```

Then `cloud_train.py run`. Do not create a second instance to retry SSH; unlock the key first, then rerun.

---

## One-command training

From the **repository root**, in the same terminal where `ssh-add -l` already lists a key:

```bash
# Smoke on a cheap GPU (no W&B)
python3 scripts/jarvislabs/cloud_train.py run --gpu L4 \
  --config configs/masked_ppo/smoke.yaml \
  --set tracking.mode=disabled

python3 scripts/jarvislabs/cloud_train.py run --gpu L4 \
  --config configs/masked_dqn/smoke.yaml \
  --set tracking.mode=disabled

# Reference run (requires WANDB_API_KEY)
python3 scripts/jarvislabs/cloud_train.py run --gpu L4 \
  --config configs/masked_ppo/reference.yaml

python3 scripts/jarvislabs/cloud_train.py run --gpu L4 \
  --config configs/masked_dqn/reference.yaml

# CPU VM
python3 scripts/jarvislabs/cloud_train.py run --cpu --vcpus 8 --ram 32 \
  --config configs/masked_ppo/dev.yaml
```

The `run` command:

1. Creates a pytorch GPU container or a CPU VM (or resumes `--on MACHINE_ID`).
2. Uploads `engine/`, `configs/`, seed YAMLs under `notebooks/`, and `scripts/jarvislabs/remote_setup.sh`.
3. Installs Python 3.12, `aresim[rllib]`, and CUDA torch when `nvidia-smi` is present.
4. Starts `aresim-rl train` in the background (`jl run`) and follows logs.
5. Downloads the remote `results/` directory into local `results/`.
6. Pauses the instance so compute billing stops (storage billing continues).

Ctrl+C while logs are streaming **detaches the laptop**, it does not download. After a finished run (including leftover W&B log spam), `run` now checks status and still fetches. If training is still going, it prints `results were NOT downloaded` and you must `fetch` before destroy/pause-delete.

The only success line for artifacts is `downloaded results to ...`. Do not delete the instance until you see that.

```bash
# Start and return immediately
python3 scripts/jarvislabs/cloud_train.py run --gpu L4 --detach

# Later
python3 scripts/jarvislabs/cloud_train.py status
python3 scripts/jarvislabs/cloud_train.py fetch
python3 scripts/jarvislabs/cloud_train.py down          # pause
python3 scripts/jarvislabs/cloud_train.py down --destroy
```

Machine id and run id are saved at `results/.jarvislabs_session.json` (gitignored with `results/`). Pass `--on MACHINE_ID` if that file is missing.

Reuse an instance without reinstalling the venv:

```bash
python3 scripts/jarvislabs/cloud_train.py run --on MACHINE_ID --gpu L4 --skip-setup \
  --config configs/masked_ppo/reference.yaml
```

`--gpu` on a reused instance still sets `resources.gpus_per_learner=1` unless you pass `--set resources.gpus_per_learner=...` yourself. Extra `--set` flags are forwarded to `aresim-rl train` unchanged.

| Flag | Purpose |
|---|---|
| `--gpu L4` | Create a pytorch container with this GPU type (`jl gpus` for availability) |
| `--cpu` | Create a CPU VM (`--vcpus`, `--ram`) |
| `--storage 100` | Disk in GB (100 is the default; VM minimum is 100) |
| `--name aresim-train` | Instance name |
| `--region IN2` | Pin IN1, IN2, or EU1 |
| `--keep` | Leave the instance running after a completed `run` |
| `--destroy` | Destroy instead of pause when `run` finishes |
| `--resume-from PATH` | Forwarded to `aresim-rl train` (remote path, usually the run directory) |

---

## Manual `jl` workflow

Use this when you want SSH, a shared filesystem, or to see each step. Replace `MACHINE_ID` with the id from `jl create` / `jl list`. `ssh-add -l` must already list a key (see [One-time setup](#one-time-setup)).

### 1. Create an instance

```bash
# GPU container (recommended). Blocks until status is Running.
jl create --gpu L4 --storage 100 --template pytorch --name aresim-train --yes --json

# CPU VM (same SSH key + ssh-add as GPU; see One-time setup)
jl create --vm --cpu --vcpus 8 --ram 32 --storage 100 --name aresim-cpu --yes --json
```

```bash
jl list
jl get MACHINE_ID
```

### 2. Upload code and seed manifests

Training needs `engine/`, `configs/`, and the seed YAMLs (evaluation fails without them). Skip `web/`, local `.venv`, and local `results/` unless you are resuming a previous cloud run.

The helper stages that payload; doing it by hand:

```bash
# From the repository root, after a local stage or by uploading trees directly:
jl upload MACHINE_ID ./engine /home/aresim/engine
jl upload MACHINE_ID ./configs /home/aresim/configs
jl upload MACHINE_ID ./notebooks/phase1_open_exploration_split_v1.yaml \
  /home/aresim/notebooks/phase1_open_exploration_split_v1.yaml
jl upload MACHINE_ID ./notebooks/phase1_smoke_eval_v1.yaml \
  /home/aresim/notebooks/phase1_smoke_eval_v1.yaml
jl upload MACHINE_ID ./scripts/jarvislabs/remote_setup.sh \
  /home/aresim/scripts/jarvislabs/remote_setup.sh
```

On CPU VMs the home directory is `/home/<user>/`, not `/home/`. Check with:

```bash
jl exec MACHINE_ID -- sh -lc 'printf %s "$HOME"'
```

### 3. Set up the environment

```bash
jl exec MACHINE_ID -- sh -lc 'chmod +x /home/aresim/scripts/jarvislabs/remote_setup.sh && /home/aresim/scripts/jarvislabs/remote_setup.sh'
```

`remote_setup.sh` installs `uv` if needed, creates `/home/aresim/.venv` with Python 3.12, then `uv pip install -e './engine[rllib]'` from PyPI (the same extra as a laptop). It does **not** use a separate JarvisLabs requirements file, and it does not pin torch to the `cu128` extra index (that index lags the `torch>=2.13` pin in `engine/pyproject.toml`). The first install is slow because of Ray.

W&B for `mode: online`:

```bash
# Writes WANDB_API_KEY on the instance; prefer a file over putting the key on the command line
printf 'WANDB_API_KEY=%s\n' "$WANDB_API_KEY" > /tmp/aresim-wandb.env
jl upload MACHINE_ID /tmp/aresim-wandb.env /home/aresim/.wandb.env
rm /tmp/aresim-wandb.env
```

### 4. Run training

Work from the remote project root so seed manifests and `results/` resolve the same way as a local checkout:

```bash
jl run --on MACHINE_ID --no-follow --yes --json -- bash -lc '
  set -euo pipefail
  cd /home/aresim
  if [ -f .wandb.env ]; then set -a; . ./.wandb.env; set +a; fi
  exec .venv/bin/aresim-rl train configs/masked_ppo/reference.yaml
'
```

Then:

```bash
jl run logs RUN_ID --follow          # Ctrl+C detaches; the job keeps running
jl run status RUN_ID
jl exec MACHINE_ID -- nvidia-smi    # GPU sanity check
```

Checked-in `reference.yaml` is sized for a 32-vCPU L4: `num_env_runners: 28`, `num_envs_per_env_runner: 1`, `gpus_per_learner: 1.0`. Training fails before Ray starts if CUDA is missing. On a CPU VM, disable the GPU learner:

```bash
--set resources.gpus_per_learner=0 --set resources.num_env_runners=28
```

### 5. Download the output folder

The run directory layout is unchanged: `results/<experiment_id>/<trial_id>/` with checkpoints, `manifest.json`, evaluation, and local W&B files. Download the whole tree:

```bash
jl download MACHINE_ID /home/aresim/results ./results -r
```

If the instance is paused, resume it first (`jl resume MACHINE_ID`); `jl download` only works while the instance is running.

### 6. Pause or destroy

```bash
jl pause MACHINE_ID --yes     # stops compute billing; disk remains
jl destroy MACHINE_ID --yes   # deletes the instance and everything on it
```

Pause when you might resume training or still need to download. Destroy when `results/` is already on your laptop and you do not need the disk. Filesystems (`jl filesystem`) survive destroy if you attached one at create time; this workflow keeps artifacts in instance home and downloads them instead.

---

## GPU vs CPU

| | GPU container | CPU VM |
|---|---|---|
| Create | `jl create --gpu L4 --template pytorch` | `jl create --vm --cpu --vcpus 8 --ram 32` |
| SSH key | Required for `jl upload` (`ssh-add` first) | Required (`jl ssh-key add` + `ssh-add`) |
| Python | Template is often 3.10; helper installs 3.12 | Same 3.12 venv via `uv` |
| Torch | CUDA wheels from the pytorch index | CPU wheels from PyPI |
| Typical `--set` | none for `reference.yaml` | `--set resources.gpus_per_learner=0` |

Pick a GPU with `jl gpus`. IN2 has L4, A100, A100-80GB, A30, H100, H200. EU1 is H100/H200 only and has a 100 GB storage minimum. Resume is region-locked and may return a **new** machine id — use the id the command prints.

---

## What gets uploaded

| Included | Not uploaded |
|---|---|
| `engine/` (no `.venv` or `__pycache__`) | `web/`, `docs/`, `.git/` |
| `configs/` | local `results/` |
| `notebooks/*.yaml` seed manifests | notebooks `.ipynb` (not needed for CLI train) |
| `scripts/jarvislabs/remote_setup.sh` | W&B keys in git (copied separately as `.wandb.env`) |

Seed YAMLs live under `notebooks/` locally. Training cannot start without them. Confirm with:

```bash
python3 scripts/jarvislabs/cloud_train.py self-check
```

---

## After the run

Same commands as local training, pointed at the downloaded run directory:

```bash
engine/.venv/bin/aresim-rl inspect results/rllib_masked_ppo_reference/seed_7
engine/.venv/bin/aresim-rl evaluate results/rllib_masked_ppo_reference/seed_7 \
  --checkpoint final --split validation
```

Resume on the cloud instance (directory already exists remotely; `artifacts.reject_existing` must allow reuse):

```bash
python3 scripts/jarvislabs/cloud_train.py run --on MACHINE_ID --gpu L4 --skip-setup \
  --config configs/masked_ppo/reference.yaml \
  --resume-from results/rllib_masked_ppo_reference/seed_7
```

`--resume-from` is a **remote** path relative to `$HOME/aresim`.

### Cost

Running instances bill compute. Paused instances bill storage only. Destroy stops both. If you detach or your laptop sleeps, the instance stays running until you `down` / `jl pause`.

---

## Troubleshooting

| Symptom | What to do |
|---|---|
| `sh: 1: set: Illegal option -o pipefail` | Ubuntu `/bin/sh` is dash. Training must run under `bash -lc`. Re-run with `--on MACHINE_ID --skip-setup` after pulling this helper. `jl run logs --follow` will sit on a dead run until you Ctrl+C. |
| `torch>=2.13` unsatisfiable / only `torch<=2.11.0+cu128` | The CUDA wheel index was consulted first. Setup now installs `engine[rllib]` from PyPI only. Re-upload and re-run setup (`run --on MACHINE_ID` without `--skip-setup`). Do not add a second requirements file. |
| `Permission denied (publickey)` / `scp: Connection closed` | JarvisLabs has the pubkey, but this shell does not. `ssh-add ~/.ssh/id_rsa` (or `id_ed25519`), then `ssh-add -l`. Do not `run --gpu` again until the agent lists a key — the failed instance may already be paused or gone. |
| `Invalid value for 'pubkey_file': ... id_ed25519.pub` | That file is missing. `ls ~/.ssh/*.pub` and add the key you have (`id_rsa.pub` on this machine). |
| `The agent has no identities` | New terminal, reboot, or expired macOS keychain unlock. Run `ssh-add` again; it does not persist across sessions unless you configure that yourself. |
| CUDA requested but unavailable | `reference.yaml` sets `gpus_per_learner: 1.0`. Use an L4 (or `--set resources.gpus_per_learner=0` on a CPU VM). |
| W&B authentication failed | `WANDB_API_KEY` missing before `run`. Export it, or put it in `.env.local` as `WANDB_API_KEY=` (not `WAND_API_KEY`). Smoke configs can use `--set tracking.mode=disabled`. |
| Seed manifest does not exist | Upload the YAML files under `notebooks/`; `self-check` lists the required names. |
| `jl download` cannot find `results/` | Training never created the directory (failed in setup). `jl run logs RUN_ID --tail 80`. |
| Resume returned a new id | Use the new `machine_id`. Update `--on` and `results/.jarvislabs_session.json`. |
| First setup takes a long time | Expected: Ray + torch install. Later `--skip-setup` reuses `/home/aresim/.venv`. |
| Logs freeze after W&B `Synced` / RLModule deprecation | Tune is done; frozen eval is running. Current builds print `Frozen evaluation N/M` and use all CPUs. An already-started sequential eval will not pick that up — stop it and run `aresim-rl evaluate` after uploading this engine. |
