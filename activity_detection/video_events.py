import argparse
import json
from pathlib import Path

from activity_detection.bed_events import detect_bed_events


def main():
    parser = argparse.ArgumentParser(
        description="Detect bed events from reviewed video observations."
    )
    parser.add_argument(
        "predictions",
        nargs="?",
        default="outputs/video_review_01/observations.json",
    )
    args = parser.parse_args()

    input_path = Path(args.predictions)

    with input_path.open(encoding="utf-8") as file:
        source = json.load(file)

    observations = source["observations"]

    if not observations:
        raise ValueError("No observations found.")

    evidence_points = []
    movement_evidence = {}
    available_times = {}

    previous_time = None
    history_available_at = 0.0

    for observation in observations:
        timestamp = observation["time_sec"]

        if previous_time is not None and timestamp <= previous_time:
            raise ValueError("Observation times must strictly increase.")

        previous_time = timestamp

        # Represent a sampled observation, not a duration interval.
        # This makes the existing detector confirm at its evidence time.
        evidence_points.append({
            "start_sec": timestamp,
            "end_sec": timestamp,
            "state": observation["state"],
        })

        movement_evidence[timestamp] = {
            "moving_away": observation.get("moving_away") is True,
            "approaching_bed": (
                observation.get("approaching_bed") is True
            ),
        }

        # Preserve the availability of all history used so far.
        history_available_at = max(
            history_available_at,
            timestamp,
            observation.get("available_at_sec", timestamp),
        )
        available_times[timestamp] = history_available_at

    events = detect_bed_events(
        evidence_points,
        movement_evidence,
    )

    for event in events:
        supporting_time = event["confirmed_sec"]

        event["supporting_observation_sec"] = supporting_time
        event["confirmed_sec"] = available_times[supporting_time]

    exit_count = sum(
        event["event"] == "bed_exit" for event in events
    )
    return_count = sum(
        event["event"] == "return_to_bed" for event in events
    )

    output = {
        "data_source": "reviewed_video_model_predictions",
        "source_video": source["source_video"],
        "observation_duration_sec": source["observation_duration_sec"],
        "confirmation_policy": (
            "Sampled evidence time, delayed when necessary until "
            "the supporting reviewed history is available."
        ),
        "movement_policy": (
            "Only an explicit true value supports directional movement. "
            "Missing or null evidence cannot confirm movement."
        ),
        "bed_exit_count": exit_count,
        "bed_return_count": return_count,
        "events": events,
    }

    output_path = input_path.parent / "bed_events.json"
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\nMODEL-PREDICTED BED EVENTS")

    if not events:
        print("No bed exit or return events detected.")

    for event in events:
        print(
            f"{event['event'].upper()}: "
            f"started at {event['start_sec']:.2f}s, "
            f"confirmed at {event['confirmed_sec']:.2f}s"
        )
        print(f"  Evidence: {event['evidence']}")

    print(f"\nBed exits: {exit_count}")
    print(f"Bed returns: {return_count}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
