"""Check versioned release files without printing potentially sensitive values."""

import argparse
import re
import subprocess
from pathlib import Path

FORBIDDEN_NAMES = {
    "AGENTS.md",
    "CLAUDE.md",
    ".env",
    ".envrc",
    "id_rsa",
    "id_ed25519",
    "handoff.md",
    "handover.md",
}
FORBIDDEN_PARTS = {
    ".claude",
    ".codex",
    ".serena",
    ".git-tunnel",
    ".intranet-tunnel",
    ".ipynb_checkpoints",
    "outputs",
    "checkpoints",
    "logs",
    ".venv",
}
PATTERNS = {
    "machine-specific path": re.compile(
        r"/(?:efs|asset|threed-code|local-ssd|home|root|Users|workspace|mnt)/"
    ),
    "internal address": re.compile(
        r"git\.woa\.com|mirrors\.tencent\.com|(?:127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "access token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}|(?:AKIA|ASIA)[A-Z0-9]{16}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{30,})\b"
    ),
}


def inspect_file(name, data):
    path = Path(name)
    problems = []
    if path.name in FORBIDDEN_NAMES or any(
        part in FORBIDDEN_PARTS for part in path.parts
    ):
        problems.append((name, 0, "private or generated file"))
    if path.suffix in {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".mp4",
        ".wav",
        ".ipynb",
    }:
        problems.append((name, 0, "generated binary artifact"))
    if len(data) > 20 * 1024 * 1024:
        problems.append((name, 0, "oversized source file"))
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        problems.append((name, 0, "unexpected binary file"))
        return problems
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in PATTERNS.items():
            if label == "internal address" and re.fullmatch(
                r"\s*[A-Za-z0-9_.-]+(?:\[[^]]+\])?==\d+(?:\.\d+)+(?:[+A-Za-z0-9_.-]*)\s*",
                line,
            ):
                continue
            if pattern.search(line):
                problems.append((name, number, label))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged", action="store_true", help="Read the exact Git index"
    )
    args = parser.parse_args()
    location = Path(__file__).resolve().parents[1]
    root = Path(
        subprocess.check_output(
            ["git", "-C", str(location), "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    command = ["git", "-C", str(root), "ls-files", "-z"]
    if not args.staged:
        command += ["--cached", "--others", "--exclude-standard"]
    names = subprocess.check_output(command).decode().split("\0")
    modes = {}
    if args.staged:
        for entry in (
            subprocess.check_output(
                ["git", "-C", str(root), "ls-files", "--stage", "-z"]
            )
            .decode()
            .split("\0")
        ):
            if entry:
                mode, name = entry.split("\t", 1)
                modes[name] = mode.split()[0]
    problems = []
    scanned = 0
    for name in sorted(set(names) - {""}):
        path = root / name
        if args.staged:
            if modes.get(name) not in {"100644", "100755"}:
                problems.append((name, 0, "non-regular Git entry requires review"))
                continue
            data = subprocess.check_output(["git", "-C", str(root), "show", f":{name}"])
        else:
            if not path.exists():
                continue
            if path.is_symlink() or root not in path.resolve().parents:
                problems.append((name, 0, "symlink requires review"))
                continue
            data = path.read_bytes()
        problems += inspect_file(name, data)
        scanned += 1
    for name, number, label in problems:
        print(f"{name}:{number}: {label}")
    if problems:
        raise SystemExit(1)
    print(f"Release source check passed: {scanned} files scanned.")


if __name__ == "__main__":
    main()
