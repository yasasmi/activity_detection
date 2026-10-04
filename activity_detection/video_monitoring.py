import argparse
import json
import math
from pathlib import Path


def generate_monitoring(observations, events, threshold):
    exits = sorted(
        [event for event in events if event["event"] == "bed_exit"],
        key=lambda event: event["supporting_observation_sec"],
    )

    results = []
    exit_index = 0
    active_exit = None
    outside_since = None
    previous_status = None
    history_available_at = 0.0

    for observation in observations:
        timestamp = observation["time_sec"]
        state = observation["state"]
        status = observation["bed_status"]

        history_available_at = max(
            history_available_at,
            timestamp,
            observation.get("available_at_sec", timestamp),
        )

        # Apply an exit only at or after its supporting observation.
        while (
            exit_index < len(exits)
            and exits[exit_index]["supporting_observation_sec"] <= timestamp
        ):
            active_exit = exits[exit_index]
            history_available_at = max(
                history_available_at,
                active_exit["confirmed_sec"],
            )
            exit_index += 1

        if status == "IN_BED":
            outside_since = None
            active_exit = None

        elif status == "UNKNOWN":
            # Continuity is unknown, but an earlier exit is not erased.
            outside_since = None

        elif status == "OUT_OF_BED":
            if previous_status != "OUT_OF_BED":
                outside_since = timestamp

        else:
            raise ValueError(f"Invalid bed status: {status}")

        continuous_outside = (
            timestamp - outside_since
            if outside_since is not None
            else 0.0
        )

        if state == "UNKNOWN" or status == "UNKNOWN":
            decision = "MONITOR"
            reason = "Activity or bed occupancy is unresolved."

            if active_exit is not None:
                reason += " A previous confirmed exit remains unresolved."

        elif status == "IN_BED":
            decision = "NORMAL"
            reason = "Known activity with the person observed in bed."

        elif active_exit is None:
            decision = "MONITOR"
            reason = (
                "Person is outside the bed, but a meaningful exit "
                "has not been confirmed."
            )

        elif continuous_outside >= threshold:
            decision = "ALERT"
            reason = (
                "A bed exit was confirmed and the continuous "
                "predicted out-of-bed duration reached the threshold."
            )

        else:
            decision = "MONITOR"
            reason = (
                "A bed exit was confirmed; continuous predicted "
                "out-of-bed duration is below the threshold."
            )

        results.append({
            "observation_time_sec": timestamp,
            "decision_available_at_sec": history_available_at,
            "state": state,
            "bed_status": status,
            "unresolved_confirmed_exit": active_exit is not None,
            "continuous_predicted_out_of_bed_sec": continuous_outside,
            "decision": decision,
            "reason": reason,
        })

        previous_status = status

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Apply monitoring rules to reviewed video predictions."
    )
    parser.add_argument(
        "--folder",
        default="outputs/video_review_01",
    )
    parser.add_argument(
        "--absence-alert-sec",
        type=float,
        default=180.0,
    )
    args = parser.parse_args()

    threshold = args.absence_alert_sec

    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("The alert threshold must be positive and finite.")

    folder = Path(args.folder)

    source = json.loads(
        (folder / "observations.json").read_text(encoding="utf-8")
    )
    event_output = json.loads(
        (folder / "bed_events.json").read_text(encoding="utf-8")
    )

    if source["source_video"] != event_output["source_video"]:
        raise ValueError("Observations and events refer to different videos.")

    decisions = generate_monitoring(
        source["observations"],
        event_output["events"],
        threshold,
    )

    output = {
        "data_source": "reviewed_video_model_predictions",
        "source_video": source["source_video"],
        "mode": "offline_retrospective_monitoring",
        "absence_alert_threshold_sec": threshold,
        "unknown_policy": (
            "Reset continuous occupancy timer; retain unresolved exit."
        ),
        "timing_policy": (
            "Decisions describe sampled observation times. "
            "Availability includes any future context used. "
            "No absence time is added solely because review is delayed."
        ),
        "decisions": decisions,
    }

    output_path = folder / "monitoring.json"
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\nMONITORING DECISION CHANGES")
    previous = None

    for result in decisions:
        current = (result["decision"], result["reason"])

        if current != previous:
            print(
                f"At {result['observation_time_sec']:.2f}s: "
                f"{result['decision']}"
            )
            print(f"  Reason: {result['reason']}")
            previous = current

    print("\nDECISION COUNTS")
    for label in ("NORMAL", "MONITOR", "ALERT"):
        count = sum(item["decision"] == label for item in decisions)
        print(f"{label}: {count}")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
