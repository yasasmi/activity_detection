import json
from pathlib import Path


# Mapping synthetic example.
ACTIVITY_TO_BED_STATUS = {
    "LYING_IN_BED": "IN_BED",
    "SITTING_ON_BED": "IN_BED",
    "SITTING_OUTSIDE_BED": "OUT_OF_BED",
    "STANDING": "OUT_OF_BED",
    "WALKING": "OUT_OF_BED",
    "OUT_OF_BED": "OUT_OF_BED",
    "UNKNOWN": "UNKNOWN",
}


def calculate_bed_occupancy(timeline, observation_duration):
    """Calculate bed occupancy from a complete activity timeline."""

    totals = {
        "IN_BED": 0,
        "OUT_OF_BED": 0,
        "UNKNOWN": 0,
    }

    occupancy_timeline = []
    current_out_of_bed_duration = 0
    longest_out_of_bed_duration = 0
    previous_end = 0

    for interval in timeline:
        start = interval["start_sec"]
        end = interval["end_sec"]
        state = interval["state"]

        if state not in ACTIVITY_TO_BED_STATUS:
            raise ValueError(f"Unexpected activity state: {state}")

        if abs(start - previous_end) > 0.000001:
            raise ValueError("Timeline contains a gap or overlap.")

        if end <= start:
            raise ValueError("Each interval must have a positive duration.")

        duration = end - start
        bed_status = ACTIVITY_TO_BED_STATUS[state]

        totals[bed_status] += duration

        # Merge adjacent intervals same bed occupancy.
        if (
            occupancy_timeline
            and occupancy_timeline[-1]["bed_status"] == bed_status
        ):
            occupancy_timeline[-1]["end_sec"] = end
        else:
            occupancy_timeline.append({
                "start_sec": start,
                "end_sec": end,
                "bed_status": bed_status,
            })

        # Count uninterrupted, known out_of_bed time.
        if bed_status == "OUT_OF_BED":
            current_out_of_bed_duration += duration
            longest_out_of_bed_duration = max(
                longest_out_of_bed_duration,
                current_out_of_bed_duration,
            )
        else:
            # In bed or unknown occupancy breaks the observed run.
            current_out_of_bed_duration = 0

        previous_end = end

    if abs(previous_end - observation_duration) > 0.000001:
        raise ValueError("Timeline does not cover the observation duration.")

    if abs(sum(totals.values()) - observation_duration) > 0.000001:
        raise ValueError("Bed occupancy totals do not match the duration.")

    return {
        "total_in_bed_sec": totals["IN_BED"],
        "total_out_of_bed_sec": totals["OUT_OF_BED"],
        "unknown_bed_duration_sec": totals["UNKNOWN"],
        "longest_observed_out_of_bed_period_sec": (
            longest_out_of_bed_duration
        ),
        "bed_occupancy_timeline": occupancy_timeline,
    }


def main():
    input_path = Path("outputs/synthetic_timeline.json")

    if not input_path.exists():
        raise FileNotFoundError(
            "Run python activity_timeline.py first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        activity_output = json.load(file)

    summary = calculate_bed_occupancy(
        activity_output["timeline"],
        activity_output["observation_duration_sec"],
    )

    print("\nSYNTHETIC TEST — no video was analyzed")

    print("\nBED OCCUPANCY TIMELINE")
    for interval in summary["bed_occupancy_timeline"]:
        print(
            f"{interval['start_sec']:>3}s to "
            f"{interval['end_sec']:>3}s: "
            f"{interval['bed_status']}"
        )

    print("\nBED OCCUPANCY SUMMARY")
    print(f"In bed: {summary['total_in_bed_sec']} seconds")
    print(f"Out of bed: {summary['total_out_of_bed_sec']} seconds")
    print(f"Unknown: {summary['unknown_bed_duration_sec']} seconds")
    print(
        "Longest observed continuous out-of-bed period: "
        f"{summary['longest_observed_out_of_bed_period_sec']} seconds"
    )

    output = {
        "data_source": "synthetic_test_not_model_predictions",
        "observation_duration_sec": (
            activity_output["observation_duration_sec"]
        ),
        **summary,
    }

    output_path = Path("outputs/synthetic_bed_summary.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
