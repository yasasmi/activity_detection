import argparse
import json
import math
from pathlib import Path


def summarize_occupancy(observations, duration):
    allowed = ("IN_BED", "OUT_OF_BED", "UNKNOWN")

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration must be positive and finite.")

    if not observations:
        raise ValueError("No observations found.")

    timestamps = [item["time_sec"] for item in observations]

    if timestamps[0] != 0:
        raise ValueError("The first observation must start at zero.")

    for index, timestamp in enumerate(timestamps):
        if not math.isfinite(timestamp):
            raise ValueError("Timestamps must be finite.")

        if not 0 <= timestamp < duration:
            raise ValueError("Observation timestamp is outside the video.")

        if index > 0 and timestamp <= timestamps[index - 1]:
            raise ValueError("Timestamps must be strictly increasing.")

    timeline = []
    durations = {status: 0.0 for status in allowed}

    for index, observation in enumerate(observations):
        status = observation["bed_status"]

        if status not in allowed:
            raise ValueError(f"Invalid bed status: {status}")

        start = timestamps[index]
        end = (
            timestamps[index + 1]
            if index + 1 < len(timestamps)
            else duration
        )

        durations[status] += end - start

        if timeline and timeline[-1]["bed_status"] == status:
            timeline[-1]["end_sec"] = end
        else:
            timeline.append({
                "start_sec": start,
                "end_sec": end,
                "bed_status": status,
            })

    longest_out = max(
        (
            item["end_sec"] - item["start_sec"]
            for item in timeline
            if item["bed_status"] == "OUT_OF_BED"
        ),
        default=0.0,
    )

    if not math.isclose(
        sum(durations.values()), duration, rel_tol=0, abs_tol=1e-6
    ):
        raise ValueError("Occupancy durations do not match video duration.")

    return timeline, durations, longest_out


def main():
    parser = argparse.ArgumentParser(
        description="Summarize predicted bed occupancy."
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

    timeline, durations, longest_out = summarize_occupancy(
        predictions["observations"],
        duration,
    )

    output = {
        "data_source": "video_model_predictions",
        "source_video": predictions["source_video"],
        "observation_duration_sec": duration,
        "duration_policy": (
            "Hold each bed-status label until the next sample; "
            "hold the final label until the analyzed video ends. "
            "UNKNOWN breaks continuous out-of-bed intervals."
        ),
        "timeline": timeline,
        "bed_duration_sec": durations,
        "longest_predicted_continuous_out_of_bed_sec": longest_out,
    }

    output_path = input_path.parent / "bed_summary.json"
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\nMODEL-PREDICTED BED OCCUPANCY")
    for item in timeline:
        print(
            f"{item['start_sec']:.2f}s to "
            f"{item['end_sec']:.2f}s: {item['bed_status']}"
        )

    print("\nESTIMATED BED DURATIONS")
    for status, seconds in durations.items():
        print(f"{status}: {seconds:.2f} seconds")

    print(
        "\nLongest predicted continuous out-of-bed period: "
        f"{longest_out:.2f} seconds"
    )
    print(f"Total accounted time: {sum(durations.values()):.2f} seconds")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
