from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2


def get_video_info(video_path: Path) -> tuple[int, float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open driving video: {video_path}")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if frames <= 0 or fps <= 0:
        raise RuntimeError("Could not read frame count/FPS from driving video.")

    return frames, fps, frames / fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PersonaLive offline inference with automatic frame handling.")
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--driving-video", required=True, type=Path)
    parser.add_argument("--python", default="python", help="Python executable used to run inference_offline.py")
    parser.add_argument("--repo-dir", default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument("--output", default=None, type=Path, help="Optional destination for the clean generated video")
    args = parser.parse_args()

    repo_dir = args.repo_dir.resolve()
    reference_image = args.reference_image.resolve()
    driving_video = args.driving_video.resolve()

    if not reference_image.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_image}")
    if not driving_video.exists():
        raise FileNotFoundError(f"Driving video not found: {driving_video}")

    frames, fps, duration = get_video_info(driving_video)
    frames_to_generate = (frames // 4) * 4
    if frames_to_generate <= 0:
        raise RuntimeError("Driving video must contain at least 4 frames.")

    print(f"Input frames: {frames}")
    print(f"Input FPS: {fps:.2f}")
    print(f"Input duration: {duration:.2f}s")
    print(f"Generating: {frames_to_generate} frames")

    run_name = "run_" + datetime.now().strftime("%H%M%S")

    cmd = [
        args.python,
        "inference_offline.py",
        "--name",
        run_name,
        "--reference_image",
        str(reference_image),
        "--driving_video",
        str(driving_video),
        "-L",
        str(frames_to_generate),
    ]

    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(cmd, cwd=repo_dir, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"PersonaLive failed with return code {result.returncode}")

    split_matches = sorted(
        glob.glob(str(repo_dir / "results" / f"*--{run_name}" / "split_vid" / "*.mp4"))
    )
    if not split_matches:
        raise FileNotFoundError("PersonaLive finished, but no clean split video was found.")

    generated_video = Path(split_matches[-1])

    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_video, output)
        generated_video = output

    print("\nPersonaLive generation complete.")
    print(f"Generated video: {generated_video}")


if __name__ == "__main__":
    main()
