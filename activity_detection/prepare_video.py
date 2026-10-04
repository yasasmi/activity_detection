import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout


def main():
    parser = argparse.ArgumentParser(
        description="Extract timestamped frames from a video."
    )
    parser.add_argument("video_path")
    parser.add_argument("--out", default="outputs/video_frames")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--limit-sec", type=float)
    args = parser.parse_args()

    video_path = Path(args.video_path).resolve()
    output_folder = Path(args.out)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--fps must be a positive number.")

    if args.limit_sec is not None:
        if not math.isfinite(args.limit_sec) or args.limit_sec <= 0:
            raise ValueError("--limit-sec must be positive.")

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} was not found.")

    # Read metadata for the first video stream.
    metadata = json.loads(run_command([
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,duration:format=duration",
        "-of", "json",
        str(video_path),
    ]))

    if not metadata.get("streams"):
        raise ValueError("The file contains no video stream.")

    stream = metadata["streams"][0]
    duration_value = stream.get("duration")

    if duration_value in (None, "N/A"):
        duration_value = metadata.get("format", {}).get("duration")

    if duration_value in (None, "N/A"):
        raise ValueError("Could not read the video duration.")

    source_duration = float(duration_value)

    if not math.isfinite(source_duration) or source_duration <= 0:
        raise ValueError("Invalid video duration.")

    analyzed_duration = source_duration

    if args.limit_sec is not None:
        analyzed_duration = min(source_duration, args.limit_sec)

    # Prevent frames from different runs from being mixed together.
    if output_folder.exists() and any(output_folder.iterdir()):
        raise ValueError(
            "Output folder is not empty. Choose a new --out folder."
        )

    output_folder.mkdir(parents=True, exist_ok=True)

    # Normalize video time to zero and sample a regular output grid.
    filters = (
        "setpts=PTS-STARTPTS,"
        f"fps=fps={args.fps}:start_time=0,"
        "scale=640:640:force_original_aspect_ratio=decrease"
    )

    print("Extracting frames...", flush=True)

    run_command([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-n",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-an",
        "-t", str(analyzed_duration),
        "-vf", filters,
        "-q:v", "2",
        str(output_folder / "frame_%06d.jpg"),
    ])

    frame_paths = sorted(output_folder.glob("frame_*.jpg"))

    if not frame_paths:
        raise RuntimeError("No frames were extracted.")

    frames = []

    for index, path in enumerate(frame_paths):
        timestamp = index / args.fps

        if timestamp < analyzed_duration:
            frames.append({
                "time_sec": timestamp,
                "file": path.name,
            })

    manifest = {
        "source_video": str(video_path),
        "source_duration_sec": source_duration,
        "analyzed_duration_sec": analyzed_duration,
        "sample_fps": args.fps,
        "timestamp_policy": "regular output grid starting at zero",
        "frame_count": len(frames),
        "frames": frames,
    }

    manifest_path = output_folder / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Source video duration: {source_duration:.2f} seconds")
    print(f"Prepared duration: {analyzed_duration:.2f} seconds")
    print(f"Sampling rate: {args.fps} frames per second")
    print(f"Frames saved: {len(frames)}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
