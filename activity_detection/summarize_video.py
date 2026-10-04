import argparse
import json
from pathlib import Path

from activity_detection.activity_timeline import build_timeline, calculate_durations


def main():
    parser = argparse.ArgumentParser(
        description="Build an activity timeline from video predictions."
    )
    parser.add_argument(
        "predictions",
        nargs="?",
        default="outputs/video_run_01/observations.json",
    )
    args = parser.parse_args()

    input_path = Path(args.predictions)

    with input_path.open(encoding="utf-8") as file:
        predictions = json.load(file)

    duration = predictions["observation_duration_sec"]

    observations = [
        {
            "time_sec": item["time_sec"],
            "state": item["state"],
        }
        for item in predictions["observations"]
    ]

    timeline = build_timeline(observations, duration)
    durations = calculate_durations(timeline)

    total = sum(durations.values())

    if abs(total - duration) > 0.000001:
        raise ValueError(
            f"Duration mismatch: accounted {total}, expected {duration}"
        )

    output = {
        "data_source": "video_model_predictions",
        "source_video": predictions["source_video"],
        "observation_duration_sec": duration,
        "duration_policy": (
            "Each sampled label is held until the next timestamp; "
            "the final label is held until the analyzed video ends."
        ),
        "timeline": timeline,
        "activity_duration_sec": durations,
        "total_accounted_sec": total,
    }

    output_path = input_path.parent / "activity_summary.json"
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\nMODEL-PREDICTED ACTIVITY TIMELINE")
    print(json.dumps(timeline, indent=2))

    print("\nESTIMATED ACTIVITY DURATIONS")
    for state, seconds in durations.items():
        print(f"{state}: {seconds:.2f} seconds")

    print(f"\nTotal accounted time: {total:.2f} seconds")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
