"""Verify a finished run and index its checkpoints and held-out videos."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio

from .checkpoint import checkpoint_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    root = args.run
    cfg = json.loads((root / "config.json").read_text())
    records = [
        json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()
    ]
    final_step = records[-1]["step"]
    stop_step = cfg.get("stop_after_steps") or cfg["max_steps"]
    if final_step != stop_step:
        raise ValueError(f"Training is incomplete: {final_step}/{stop_step}")
    checkpoint = root / f"checkpoint-{final_step:06d}"
    required = checkpoint_files(cfg.get("training_mode", "lora"))
    if any(not (checkpoint / name).is_file() for name in required):
        raise FileNotFoundError("Final checkpoint is incomplete")
    videos = []
    rows = []
    if cfg["validation_embeddings"]:
        manifest = json.loads(
            (Path(cfg["validation_embeddings"]) / "manifest.json").read_text()
        )
        for index, entry in enumerate(manifest["entries"]):
            teacher = root / "validation/teacher" / f"{index:02d}.mp4"
            student = (
                root
                / "validation/student"
                / f"step-{final_step:06d}"
                / f"{index:02d}.mp4"
            )
            for path in (teacher, student):
                with imageio.get_reader(path) as reader:
                    meta = reader.get_meta_data()
                    count = reader.count_frames()
                if (
                    count != cfg["frames"]
                    or abs(meta["fps"] - cfg["validation_fps"]) > 0.01
                    or meta["size"]
                    != (
                        cfg["width"],
                        cfg["height"],
                    )
                ):
                    raise ValueError(f"Invalid video shape or frame count: {path}")
                videos.append(
                    {
                        "path": str(path.relative_to(root)),
                        "frames": count,
                        "fps": meta["fps"],
                        "bytes": path.stat().st_size,
                    }
                )
            rows.append(
                f"| {index} | {entry['prompt'].replace('|', '/')} | [teacher]({teacher.relative_to(root)}) | [student]({student.relative_to(root)}) |"
            )
    source = json.loads((root / "source.json").read_text())
    summary = {
        "status": "complete",
        "step": final_step,
        "schedule_steps": cfg["max_steps"],
        "checkpoint": str(checkpoint),
        "source": source,
        "runtime": json.loads((root / "runtime.json").read_text()),
        "validation_sampler": cfg.get("validation_sampler", "trajectory"),
        "training_mode": cfg.get("training_mode", "lora"),
        "final_metrics": records[-1],
        "videos": videos,
        "quality_assessment": "Pending visual review; valid videos and finite losses do not establish generation quality.",
    }
    (root / "completion.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = (
        f"# {Path(cfg['model']).name} 训练完成\n\n更新数：{final_step}。源代码 commit：`{source['commit']}`。\n\n最终权重：[checkpoint]({checkpoint.name}/)。完整状态见 [completion.json](completion.json)。\n\nStudent 采样：`{cfg.get('validation_sampler', 'trajectory')}`。视频已验证帧数、分辨率和帧率；质量仍需对照提示词目视检查，训练 loss 不作为质量分数。\n\n| 编号 | 提示词 | Teacher 50 步 | Student {cfg['distill_steps']} 步 |\n| --- | --- | --- | --- |\n"
        + "\n".join(rows)
        + "\n"
    )
    (root / "REPORT.md").write_text(text)
    print(
        json.dumps(
            {
                "status": "complete",
                "step": final_step,
                "videos": len(videos),
                "report": str(root / "REPORT.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
