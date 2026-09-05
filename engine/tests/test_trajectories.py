"""Round-trip, integrity, corruption, and full-pipeline trajectory tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from aresim.defaults import DEFAULT_ENGINE_CONFIG
from aresim.gameplay import ReplayCursor, normalize_gameplay_payload
from aresim.training import (
    EpisodeSpec,
    RolloutConfig,
    RolloutRunner,
    TrajectoryValidationError,
    TrajectoryWriter,
    iter_trajectory_episodes,
    validate_trajectory_episode,
    validate_trajectory_dataset,
)
from aresim.training.trajectories import describe_space, space_from_descriptor, validate_episode


def _assert_value_equal(first, second) -> None:
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_value_equal(first[key], second[key])
        return
    if isinstance(first, tuple):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _assert_value_equal(left, right)
        return
    np.testing.assert_array_equal(first, second)


def _episode(episode_id: str = "episode-0", *, steps: int = 4):
    result = RolloutRunner(
        RolloutConfig((EpisodeSpec(episode_id, 1447, 31),), max_episode_steps=steps),
        "random_valid",
    ).run()
    return result.episodes[0]


@pytest.mark.parametrize("compression", ["none", "gzip"])
def test_jsonl_writer_round_trips_spaces_dtypes_and_transition_fields(tmp_path: Path, compression: str) -> None:
    output = tmp_path / f"dataset-{compression}"
    writer = TrajectoryWriter(output, f"test-{compression}", compression=compression, episodes_per_shard=1)
    result = RolloutRunner(
        RolloutConfig(
            (EpisodeSpec("episode-a", 1447, 31), EpisodeSpec("episode-b", 2468, 32)),
            max_episode_steps=4,
        ),
        "random_valid",
    ).run(writer)

    manifest = validate_trajectory_dataset(output)
    loaded = list(iter_trajectory_episodes(output))
    assert manifest.episode_count == 2
    assert manifest.transition_count == 8
    assert len(manifest.payload["shards"]) == 2
    expected_suffix = ".jsonl.gz" if compression == "gzip" else ".jsonl"
    assert all(str(shard["path"]).endswith(expected_suffix) for shard in manifest.payload["shards"])
    assert manifest.payload["policy_sources"] == ["aresim.agent.random_valid.v1"]
    assert manifest.payload["include_episode_artifacts"] is True
    assert len(manifest.payload["episode_artifacts"]) == 2
    assert result.artifact_manifest == str((output / "manifest.json").resolve())

    for expected, actual in zip(result.episodes, loaded, strict=True):
        assert expected.actions == actual.actions
        assert expected.rewards == actual.rewards
        assert expected.reward_breakdowns == actual.reward_breakdowns
        assert expected.engine_reward_terms == actual.engine_reward_terms
        assert expected.terminated == actual.terminated
        assert expected.truncated == actual.truncated
        assert expected.events == actual.events
        assert expected.state_checksums == actual.state_checksums
        assert expected.replay == actual.replay
        assert actual.replay is not None
        assert actual.replay["schemaVersion"] == "aresim.trajectory.replay.v1"
        assert actual.replay["metadata"]["algorithmId"] == expected.policy_id
        assert actual.replay["integrity"]["stepCount"] == expected.length
        for left, right in zip(expected.observations, actual.observations, strict=True):
            _assert_value_equal(left, right)
        assert actual.observations[0]["terrain_type"].dtype == np.uint8
        assert actual.observations[0]["spatial"].dtype == np.float32
        assert actual.action_masks[0].dtype == np.int8

    for artifact_entry in manifest.payload["episode_artifacts"]:
        artifact_path = output / artifact_entry["path"]
        artifact = validate_trajectory_episode(artifact_path)
        raw_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert raw_artifact["schemaVersion"] == "aresim.trajectory.episode.v1"
        assert raw_artifact["replay"]["schemaVersion"] == "aresim.trajectory.replay.v1"
        assert artifact["schemaVersion"] == "aresim.trajectory.episode.v1"
        assert artifact["policy"] is not None
        assert artifact["replay"]["schemaVersion"] == "aresim.trajectory.replay.v1"
        gameplay = artifact["replay"]
        cursor = ReplayCursor.create(gameplay, artifact_path.name)
        final_snapshot = cursor.jump(gameplay["integrity"]["finalStep"])
        assert final_snapshot["mode"] == "Replay"
        assert final_snapshot["step"] == gameplay["finalSnapshot"]["step"]
        assert final_snapshot["terrain"] == gameplay["finalSnapshot"]["terrain"]
        assert final_snapshot["rovers"] == gameplay["finalSnapshot"]["rovers"]


def test_space_descriptors_support_every_declared_gymnasium_family() -> None:
    supported = (
        spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        spaces.Discrete(4, start=1),
        spaces.Dict({"value": spaces.Box(0, 1, shape=(1,), dtype=np.float32)}),
        spaces.Tuple((spaces.Discrete(2), spaces.MultiBinary(2))),
        spaces.MultiBinary((2, 3)),
        spaces.MultiDiscrete([2, 3], start=[0, 1]),
    )
    for original in supported:
        restored = space_from_descriptor(describe_space(original))
        assert type(restored) is type(original)
        assert restored.shape == original.shape
    with pytest.raises(TypeError, match="unsupported"):
        describe_space(spaces.Sequence(spaces.Discrete(2)))


def test_gzip_shards_are_byte_deterministic_for_identical_episodes(tmp_path: Path) -> None:
    episode = _episode(steps=3)
    shard_bytes = []
    for name in ("first", "second"):
        output = tmp_path / name
        writer = TrajectoryWriter(output, name, compression="gzip")
        writer.write_episode(episode)
        writer.finalize(max_episode_steps=3)
        shard_bytes.append((output / "episodes-00000.jsonl.gz").read_bytes())
    assert shard_bytes[0] == shard_bytes[1]


def test_writer_rejects_existing_output_and_invalid_or_empty_datasets(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        TrajectoryWriter(existing, "dataset")
    writer = TrajectoryWriter(tmp_path / "empty", "dataset")
    with pytest.raises(ValueError, match="empty"):
        writer.finalize(max_episode_steps=10)
    invalid = replace(_episode(steps=2), rewards=(float("nan"), 0.0))
    with pytest.raises(TrajectoryValidationError, match="finite"):
        validate_episode(invalid)


def test_writer_can_explicitly_disable_standalone_episode_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "compact-only"
    writer = TrajectoryWriter(output, "compact-only", include_episode_artifacts=False)
    writer.write_episode(_episode(steps=2))
    writer.finalize(max_episode_steps=2)

    manifest = validate_trajectory_dataset(output)
    loaded = tuple(iter_trajectory_episodes(output))
    assert manifest.payload["include_episode_artifacts"] is False
    assert manifest.payload["episode_artifacts"] == []
    assert loaded[0].replay is None


def _plain_dataset(tmp_path: Path) -> Path:
    output = tmp_path / "dataset"
    writer = TrajectoryWriter(output, "corruption-test")
    writer.write_episode(_episode(steps=3))
    writer.finalize(max_episode_steps=3)
    return output


def _rewrite_shard(output: Path, mutate) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shard_path = output / manifest["shards"][0]["path"]
    payload = json.loads(shard_path.read_text().strip())
    mutate(payload)
    shard_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")


def test_validation_rejects_checksum_path_alignment_and_reward_corruption(tmp_path: Path) -> None:
    checksum_output = _plain_dataset(tmp_path / "checksum")
    manifest = json.loads((checksum_output / "manifest.json").read_text())
    shard = checksum_output / manifest["shards"][0]["path"]
    shard.write_bytes(shard.read_bytes() + b" ")
    with pytest.raises(TrajectoryValidationError, match="checksum"):
        validate_trajectory_dataset(checksum_output)

    path_output = _plain_dataset(tmp_path / "path")
    manifest_path = path_output / "manifest.json"
    path_manifest = json.loads(manifest_path.read_text())
    path_manifest["shards"][0]["path"] = "../escape.jsonl"
    manifest_path.write_text(json.dumps(path_manifest))
    with pytest.raises(TrajectoryValidationError, match="safe"):
        validate_trajectory_dataset(path_output)

    alignment_output = _plain_dataset(tmp_path / "alignment")
    _rewrite_shard(alignment_output, lambda payload: payload["observations"].pop())
    with pytest.raises(TrajectoryValidationError, match=r"T\+1"):
        validate_trajectory_dataset(alignment_output)

    reward_output = _plain_dataset(tmp_path / "reward")
    _rewrite_shard(reward_output, lambda payload: payload["rewards"].__setitem__(0, 99))
    with pytest.raises(TrajectoryValidationError, match="breakdown"):
        validate_trajectory_dataset(reward_output)

    artifact_output = _plain_dataset(tmp_path / "artifact")
    artifact_manifest = json.loads((artifact_output / "manifest.json").read_text())
    artifact_path = artifact_output / artifact_manifest["episode_artifacts"][0]["path"]
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
    with pytest.raises(TrajectoryValidationError, match="episode artifact checksum"):
        validate_trajectory_dataset(artifact_output)


def test_validation_rejects_manifest_identifier_drift(tmp_path: Path) -> None:
    output = _plain_dataset(tmp_path / "identifiers")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["policy_sources"] = ["aresim.agent.wait.v1"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(TrajectoryValidationError, match="policy_sources"):
        validate_trajectory_dataset(output)
