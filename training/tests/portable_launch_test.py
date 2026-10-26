"""Verify portable launch arguments and relative model/configuration paths."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from video_blade.config import TrainConfig
from video_blade.h3.config import H3Config


@pytest.mark.parametrize(
    "family,module", [("dense", "video_blade.train"), ("h3", "video_blade.h3.train")]
)
def test_launcher_uses_selected_python_from_any_directory(tmp_path, family, module):
    root = Path(__file__).resolve().parents[1]
    stub = tmp_path / "python-stub"
    stub.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        'print(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd()}))\n'
    )
    stub.chmod(0o755)
    environment = {**os.environ, "VIDEO_BLADE_PYTHON": str(stub), "NPROC_PER_NODE": "2"}
    result = subprocess.run(
        [
            "bash",
            str(root / "scripts" / f"train_{family}.sh"),
            "--config",
            "example.json",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(result.stdout)
    assert output["cwd"] == str(root)
    assert output["argv"] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "-m",
        module,
        "--config",
        "example.json",
    ]


def test_public_configs_validate_and_use_relative_paths():
    root = Path(__file__).resolve().parents[1]
    assert not Path(TrainConfig().model).is_absolute()
    assert not Path(H3Config().model).is_absolute()
    for path in (root / "configs").glob("*.json"):
        values = json.loads(path.read_text())
        config_type = H3Config if path.name.startswith("h3_") else TrainConfig
        cfg = config_type(**values).validate()
        for name in ("model", "embeddings", "output", "validation_embeddings"):
            assert not Path(getattr(cfg, name)).is_absolute()


def test_current_and_original_implementations_are_separate():
    current = Path(__file__).resolve().parents[1]
    repository = current.parent
    assert current.name == "training"
    assert (current / "pyproject.toml").is_file()
    assert (repository / "legacy/pyproject.toml").is_file()
    assert (repository / "legacy/cogvideox/train/train_cogvideo_tdm.py").is_file()
    assert (repository / "legacy/wanx/train/train_wanx_tdm.py").is_file()
    assert not (repository / "pyproject.toml").exists()
    assert not (repository / "video_blade").exists()
    assert (
        "working-directory: training"
        in (repository / ".github/workflows/dense-tests.yml").read_text()
    )
