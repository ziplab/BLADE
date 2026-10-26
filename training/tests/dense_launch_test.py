"""Reject duplicate runs before model loading or distributed initialization."""

import pytest

from video_blade import train
from video_blade.config import TrainConfig


def test_duplicate_output_rejected_before_model_loading(tmp_path, monkeypatch):
    (tmp_path / "metrics.jsonl").write_text('{"step": 1}\n')
    cfg = TrainConfig(model="/nonexistent/model", output=str(tmp_path))
    monkeypatch.setattr(train, "parse_config", lambda: cfg)
    with pytest.raises(FileExistsError, match="Use a new output directory or --resume"):
        train.run_training()
