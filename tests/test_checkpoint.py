from __future__ import annotations

from pathlib import Path

import pytest
import torch

from weathermesh5c.checkpoint import load_checkpoint_state, sha256_file


def test_plain_state_dictionary_loads_without_pickle_execution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "state.pt"
    expected = {"layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    torch.save(expected, checkpoint)

    loaded = load_checkpoint_state(checkpoint)

    torch.testing.assert_close(loaded["layer.weight"], expected["layer.weight"])


def test_training_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "training.pt"
    torch.save({"model_state_dict": {"layer.weight": torch.ones(1)}, "step": 7}, checkpoint)

    with pytest.raises(TypeError, match="non-tensor state"):
        load_checkpoint_state(checkpoint)


def test_sha256_detects_content_changes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"weather-mesh")
    first = sha256_file(checkpoint)
    checkpoint.write_bytes(b"weather-mesh-5c")

    assert sha256_file(checkpoint) != first
