import json
from pathlib import Path


# The seven states
VALID_STATES = (
    "LYING_IN_BED",
    "SITTING_ON_BED",
    "SITTING_OUTSIDE_BED",
    "STANDING",
    "WALKING",
    "OUT_OF_BED",
    "UNKNOWN",
)


def build_timeline(observations, video_duration):
    """Combine timestamped observations into continuous activity intervals."""

    if video_duration <= 0:
        raise ValueError("Video duration must be greater than zero.")

    if not observations:
        raise ValueError("At least one observation is required.")

    if observations[0]["time_sec"] != 0:
        raise ValueError("The first observation must start at zero.")

    previous_time = -1

    for observation in observations:
        timestamp = observation["time_sec"]
        state = observation["state"]

        if state not in VALID_STATES:
            raise ValueError(f"Invalid activity state: {state}")

        if not 0 <= timestamp < video_duration:
            raise ValueError("Observation time is outside the video.")

        if timestamp <= previous_time:
            raise ValueError("Observation times must increase strictly.")

        previous_time = timestamp

    timeline = []
    current_state = observations[0]["state"]
    interval_start = 0

    for observation in observations[1:]:
        new_state = observation["state"]
        timestamp = observation["time_sec"]

        if new_state != current_state:
            timeline.append({
                "start_sec": interval_start,
                "end_sec": timestamp,
                "state": current_state,
            })

            current_state = new_state
            interval_start = timestamp

    # Closing the final interval at the end of the video.
    timeline.append({
        "start_sec": interval_start,
        "end_sec": video_duration,
        "state": current_state,
    })

    return timeline


def calculate_durations(timeline):
    """Add up the seconds assigned to each activity."""

    durations = {state: 0 for state in VALID_STATES}

    for interval in timeline:
        seconds = interval["end_sec"] - interval["start_sec"]
        durations[interval["state"]] += seconds

    return durations


def main():
    # test inputs for the logic.
    # NOT predictions from a vision model.
    observations = [
        {"time_sec": 0, "state": "LYING_IN_BED"},
        {"time_sec": 5, "state": "LYING_IN_BED"},
        {"time_sec": 10, "state": "SITTING_ON_BED"},
        {"time_sec": 15, "state": "STANDING"},
        {"time_sec": 20, "state": "WALKING"},
        {"time_sec": 25, "state": "WALKING"},
        {"time_sec": 30, "state": "SITTING_OUTSIDE_BED"},
        {"time_sec": 40, "state": "UNKNOWN"},
        {"time_sec": 45, "state": "WALKING"},
        {"time_sec": 50, "state": "SITTING_ON_BED"},
        {"time_sec": 55, "state": "LYING_IN_BED"},
    ]

    video_duration = 60

    timeline = build_timeline(observations, video_duration)
    durations = calculate_durations(timeline)

    total = sum(durations.values())

    if abs(total - video_duration) > 0.000001:
        raise ValueError("Activity durations do not match the video duration.")

    print("\nSYNTHETIC TEST — no video was analyzed")

    print("\nACTIVITY TIMELINE")
    for interval in timeline:
        print(
            f"{interval['start_sec']:>3}s to "
            f"{interval['end_sec']:>3}s: "
            f"{interval['state']}"
        )

    print("\nACTIVITY DURATIONS")
    for state, seconds in durations.items():
        print(f"{state}: {seconds} seconds")

    print(f"\nTotal accounted time: {total} seconds")

    output = {
        "data_source": "synthetic_test_not_model_predictions",
        "observation_duration_sec": video_duration,
        "timeline": timeline,
        "activity_duration_sec": durations,
    }

    output_path = Path("outputs/synthetic_timeline.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

