"""Verify optional RLlib training, W&B metrics, and extension points.

The real learner smoke remains opt-in because it starts local Ray workers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")

from aresim.algorithms.common.tracking import load_wandb_run_id, new_wandb_run_id, resolve_wandb_run_id
from aresim.factory import make_gym_env
from aresim.algorithms.ppo.train import (
    LocalMaskedActorCritic,
    _checkpoint_label,
    _environment_steps_from_result,
    _export_checkpoint,
    _prune_exported_checkpoints,
    _wandb_callbacks,
    canonicalize_rllib_metrics,
    run_experiment,
)
from aresim.algorithms.registry import TrainingContext, create_training_registry
from aresim.training.experiments import apply_overrides, load_experiment
from aresim.training.seeds import load_seed_manifest, resolve_checkout_file


ROOT = Path(__file__).parents[1]
SMOKE = "configs/masked_ppo/smoke.yaml"


def _tensor(value):
    if isinstance(value, dict):
        return {key: _tensor(item) for key, item in value.items()}
    return torch.as_tensor(np.asarray(value)).unsqueeze(0)


def test_experiment_yaml_is_strict_typed_and_hash_stable() -> None:
    spec = load_experiment(SMOKE)
    assert spec.config_hash == load_experiment(SMOKE).config_hash
    assert spec.evaluation.seed_manifest == "notebooks/phase1_smoke_eval_v1.yaml"
    load_seed_manifest(spec.evaluation.seed_manifest)
    assert spec.model_config.conv_channels == (32, 64)
    changed = apply_overrides(spec.as_dict(), ("algorithm_config.learning_rate=0.0001",))
    assert changed.algorithm_config.learning_rate == 0.0001
    assert changed.config_hash != spec.config_hash
    with pytest.raises(ValueError, match="unknown override path"):
        apply_overrides(spec.as_dict(), ("algorithm_config.missing=1",))
    with pytest.raises(TypeError, match="type mismatch"):
        apply_overrides(spec.as_dict(), ("algorithm_config.update_epochs=true",))
    with pytest.raises(ValueError, match="unknown experiment fields"):
        apply_overrides({**spec.as_dict(), "surprise": 1}, ())


def test_seed_split_has_expected_disjoint_fixed_populations() -> None:
    manifest = load_seed_manifest("notebooks/phase1_open_exploration_split_v1.yaml")
    assert (len(manifest.train), len(manifest.validation), len(manifest.test)) == (512, 32, 100)
    assert set(manifest.train).isdisjoint(manifest.validation)
    assert set(manifest.train).isdisjoint(manifest.test)
    assert set(manifest.validation).isdisjoint(manifest.test)
    assert manifest.episodes("validation")[0].environment_seed == manifest.validation[0]


def test_local_actor_critic_masks_logits_and_handles_empty_objectives() -> None:
    spec = load_experiment(SMOKE)
    environment = make_gym_env(max_episode_steps=5)
    policy_input, _ = environment.reset(seed=1447)
    mask = policy_input["action_mask"].copy()
    assert not np.all(mask)
    model = LocalMaskedActorCritic(spec.model_config)
    logits, values = model(_tensor(policy_input["observation"]), _tensor(mask))
    assert logits.shape == (1, 10)
    assert values.shape == (1,)
    assert torch.isfinite(logits[:, mask == 1]).all()
    assert torch.all(logits[:, mask == 0] == torch.finfo(logits.dtype).min)
    assert int(torch.argmax(logits, dim=-1)[0]) in np.flatnonzero(mask)
    with pytest.raises(ValueError, match="no legal action"):
        model(_tensor(policy_input["observation"]), torch.zeros((1, 10), dtype=torch.int8))


def test_training_registry_rejects_duplicates_and_unknown_names() -> None:
    spec = load_experiment(SMOKE)
    registry = create_training_registry()
    context = TrainingContext(spec)
    assert registry.build_algorithm("masked_ppo", context).algorithm_id == "masked_ppo"
    assert registry.build_model("local_cnn_actor_critic", context).model_id == "local_cnn_actor_critic"
    with pytest.raises(ValueError, match="duplicate model"):
        registry.register_model("local_cnn_actor_critic", lambda selected: object())
    with pytest.raises(ValueError, match="unknown model"):
        registry.build_model("missing", context)


def test_wandb_run_id_helpers(tmp_path: Path) -> None:
    run_id = new_wandb_run_id()
    assert len(run_id) == 24
    assert all(character in "0123456789abcdef" for character in run_id)
    assert load_wandb_run_id(tmp_path) is None
    manifest = {"wandb_run_id": run_id}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_wandb_run_id(tmp_path) == run_id
    assert resolve_wandb_run_id(tmp_path) == run_id
    assert resolve_wandb_run_id(tmp_path, "b" * 24) == "b" * 24


def test_canonical_metrics_are_ready_for_wandb(tmp_path: Path) -> None:
    result = {
        "num_env_steps_sampled_lifetime": 64,
        "env_runners": {"episode_return_mean": 1.5, "ares/invalid_actions": 0.0},
        "learners": {"default_policy": {"total_loss": 2.0, "entropy": 0.7, "vf_explained_var": 0.2}},
    }
    metrics = canonicalize_rllib_metrics(result)
    assert metrics["train/environment_steps"] == 64
    assert metrics["train/shaped_return"] == 1.5
    assert metrics["train/invalid_actions"] == 0
    assert metrics["learner/explained_variance"] == 0.2
    manifest = {"wandb_run_id": "a" * 24}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _wandb_callbacks(load_experiment(SMOKE), tmp_path) == []
    offline = replace(load_experiment(SMOKE), tracking=replace(load_experiment(SMOKE).tracking, mode="offline"))
    assert len(_wandb_callbacks(offline, tmp_path)) == 1


def test_wandb_run_path_requires_tracked_run(tmp_path: Path) -> None:
    manifest = {
        "experiment": {"tracking": {"mode": "disabled", "project": "aresim"}},
        "wandb_run_id": "abc123",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    from aresim.training.reports import wandb_run_path

    with pytest.raises(ValueError, match="tracked W&B run"):
        wandb_run_path(tmp_path)


def test_generate_report_writes_plots_from_wandb_history(tmp_path: Path, monkeypatch) -> None:
    import pandas as pd

    from aresim.training.reports import generate_report

    manifest = {
        "experiment": {"tracking": {"mode": "online", "project": "aresim", "entity": "test-entity"}},
        "wandb_run_id": "abc123",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    evaluation = tmp_path / "evaluation" / "final"
    evaluation.mkdir(parents=True)
    summary = {
        "mean_shaped_return": 1.5,
        "baseline_comparisons": [
            {"policy_id": "aresim.agent.random_valid.v1", "mean_shaped_return": -2.0},
        ],
    }
    (evaluation / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    history = pd.DataFrame(
        {
            "train/environment_steps": [1024, 2048],
            "train/shaped_return": [0.1, 0.4],
            "train/episode_length": [100.0, 120.0],
            "learner/total_loss": [2.0, 1.5],
            "learner/entropy": [0.8, 0.7],
        }
    )

    class FakeRun:
        def history(self, samples: int = 10_000):
            return history

    class FakeApi:
        default_entity = "test-entity"

        def run(self, path: str):
            assert path == "test-entity/aresim/abc123"
            return FakeRun()

    import wandb

    monkeypatch.setattr(wandb, "Api", lambda: FakeApi())
    reports_dir = generate_report(tmp_path)
    assert reports_dir == tmp_path / "reports"
    assert (reports_dir / "training_returns.png").is_file()
    assert (reports_dir / "evaluation_baseline_comparison.png").is_file()
    assert (reports_dir / "metrics_history.json").is_file()
    report_manifest = json.loads((reports_dir / "manifest.json").read_text(encoding="utf-8"))
    assert report_manifest["wandb_run"] == "test-entity/aresim/abc123"
    assert "training_returns.png" in report_manifest["plots"]


def test_checkpoint_label_uses_zero_padded_environment_steps() -> None:
    assert _checkpoint_label(81920) == "step_081920"
    assert _checkpoint_label(1048576) == "step_1048576"


def test_environment_steps_from_result_prefers_canonical_metrics() -> None:
    result = {
        "env_runners": {"num_env_steps_sampled_lifetime": 8195.0, "episode_return_mean": -1.0},
        "training_iteration": 2,
    }
    assert _environment_steps_from_result(result, rollout_batch_size=4096) == 8195


def test_prune_exported_checkpoints_keeps_latest_step_exports(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    for label in ("step_000001", "step_000002", "step_000003", "final"):
        directory = checkpoints / label
        directory.mkdir(parents=True)
        (directory / "checkpoint.json").write_text("{}", encoding="utf-8")
    _prune_exported_checkpoints(tmp_path, keep=2)
    remaining = sorted(path.name for path in checkpoints.iterdir() if path.is_dir())
    assert remaining == ["final", "step_000002", "step_000003"]


def test_export_checkpoint_writes_ui_ready_sidecar(tmp_path: Path) -> None:
    smoke_sidecar = ROOT.parent / "results/rllib_masked_ppo_smoke/seed_7/checkpoints/final/checkpoint.json"
    if not smoke_sidecar.is_file():
        pytest.skip("smoke checkpoint is required for export regression")
    smoke_native = smoke_sidecar.parent / "native"
    spec = load_experiment(SMOKE)
    registry = create_training_registry()
    context = TrainingContext(spec)
    algorithm = registry.build_algorithm("masked_ppo", context)
    schemas = {
        "observation_schema": algorithm.observation_schema,
        "action_schema": algorithm.action_schema,
        "task_id": "phase1_open_exploration_v1",
        "reward_profile": "aresim.reward.shaped_train.v1",
    }
    from ray.train import Checkpoint

    sidecar = _export_checkpoint(
        tmp_path,
        Checkpoint.from_directory(str(smoke_native)),
        "step_004096",
        spec,
        algorithm,
        schemas,
    )
    assert sidecar == tmp_path / "checkpoints" / "step_004096" / "checkpoint.json"
    from aresim.algorithms.ppo.checkpoint import make_checkpoint_agent

    agent = make_checkpoint_agent(sidecar)
    assert agent.policy_id.endswith(":step_004096")


def test_base_package_import_does_not_require_training_dependencies() -> None:
    script = "import sys; sys.modules['ray']=None; sys.modules['torch']=None; import aresim; print(aresim.__name__)"
    completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, text=True, capture_output=True)
    assert completed.stdout.strip() == "aresim"


@pytest.mark.rllib_slow
@pytest.mark.skipif(os.environ.get("ARESIM_RUN_RLLIB_TESTS") != "1", reason="set ARESIM_RUN_RLLIB_TESTS=1 to start Ray")
def test_one_real_masked_ppo_update_and_checkpoint(tmp_path: Path) -> None:
    spec = load_experiment(SMOKE)
    spec = replace(
        spec,
        experiment_id="pytest_masked_ppo",
        trial_id="one_update",
        algorithm_config=replace(spec.algorithm_config, total_environment_steps=64, rollout_batch_size=64, minibatch_size=32, update_epochs=1),
        environment=replace(spec.environment, max_episode_steps=20),
        checkpoint=replace(spec.checkpoint, interval_environment_steps=64),
        tracking=replace(spec.tracking, mode="offline"),
        artifacts=replace(spec.artifacts, root=str(tmp_path)),
    )
    run = run_experiment(spec, evaluate=False, report=False)
    assert (run / "checkpoints" / "final" / "checkpoint.json").is_file()
    assert (run / "checkpoints" / "step_000064" / "checkpoint.json").is_file()
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert isinstance(manifest["wandb_run_id"], str) and len(manifest["wandb_run_id"]) == 24
    assert not (run / "metrics").exists()
