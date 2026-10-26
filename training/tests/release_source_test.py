"""Protect release filtering against private files, tokens, and false positives."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "release_check", Path(__file__).resolve().parents[1] / "scripts/check_release.py"
)
release_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_check)


@pytest.mark.parametrize(
    "name",
    [
        "AGENTS.md",
        ".env",
        "docs/handoff.md",
        ".codex/config.toml",
        "outputs/run/config.json",
    ],
)
def test_private_files_are_rejected(name):
    assert release_check.inspect_file(name, b"example")


@pytest.mark.parametrize(
    "value", ["hf_" + "a" * 40, "ghp_" + "a" * 40, "sk-proj-" + "a" * 40]
)
def test_access_tokens_are_rejected(value):
    assert release_check.inspect_file("config.py", value.encode())


def test_machine_paths_and_private_addresses_are_rejected():
    path = "/" + "efs" + "/" + "example-owner/model"
    address = "http://" + "10" + ".23.0.5"
    assert release_check.inspect_file("config.py", path.encode())
    assert release_check.inspect_file("config.py", address.encode())


def test_dependency_version_is_not_treated_as_an_ip_address():
    requirement = "nvidia-curand-cu12==10" + ".3.9.90\n"
    assert not release_check.inspect_file("requirements.txt", requirement.encode())


def test_public_relative_paths_are_allowed():
    assert not release_check.inspect_file(
        "configs/h3.json", b'{"model": "models/MiniMax-H3"}'
    )


@pytest.mark.parametrize("entry", ["symlink", "private-file"])
def test_staged_scan_rejects_unsafe_git_entries(tmp_path, entry):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "training/scripts").mkdir(parents=True)
    (tmp_path / "legacy").mkdir()
    script = tmp_path / "training/scripts/check_release.py"
    shutil.copyfile(Path(release_check.__file__), script)
    if entry == "symlink":
        unsafe = tmp_path / "legacy/linked.py"
        unsafe.symlink_to("missing-target.py")
    else:
        unsafe = tmp_path / "legacy/AGENTS.md"
        unsafe.write_text("Local instructions must not be published.\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", str(unsafe)], check=True)
    result = subprocess.run(
        [sys.executable, str(script), "--staged"], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert unsafe.name in result.stdout
