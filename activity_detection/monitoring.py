import json
from pathlib import Path

from activity_detection.bed_occupancy import ACTIVITY_TO_BED_STATUS


# Demonstration policy: absence this long requires an alert.
ABSENCE_ALERT_SEC = 180


def make_decision(state, bed_status, absence_seconds):
    """Return a decision and an explanation."""

    if state == "UNKNOWN" or bed_status == "UNKNOWN":
        return (
            "MONITOR",
            "Insufficient evidence to determine activity or bed occupancy.",
        )

    if absence_seconds is not None:
        if absence_seconds >= ABSENCE_ALERT_SEC:
            return (
                "ALERT",
                "Confirmed continuous absence reached the configured limit.",
            )

        return (
            "MONITOR",
            "Confirmed bed exit; monitoring continued absence.",
        )

    return (
        "NORMAL",
        "Known activity with no active monitoring condition.",
    )


def generate_decisions(timeline, events):
    """Evaluate the monitoring rules at each interval endpoint."""

    exit_events = sorted(
        [
            event
            for event in events
            if event["event"] == "bed_exit"
        ],
        key=lambda event: event["confirmed_sec"],
    )

    decisions = []
    exit_index = 0
    active_exit_start = None

    for interval in timeline:
        start = interval["start_sec"]
        end = interval["end_sec"]
        state = interval["state"]
        bed_status = ACTIVITY_TO_BED_STATUS[state]

        # Use an exit only once its confirmation time has been reached.
        while (
            exit_index < len(exit_events)
            and exit_events[exit_index]["confirmed_sec"] <= end
        ):
            active_exit_start = exit_events[exit_index]["start_sec"]
            exit_index += 1

        # Returning to bed ends the absence.
        # Unknown occupancy breaks our evidence of continuous absence.
        if bed_status != "OUT_OF_BED":
            active_exit_start = None

        absence_seconds = None

        if active_exit_start is not None:
            absence_seconds = end - active_exit_start

        decision, reason = make_decision(
            state,
            bed_status,
            absence_seconds,
        )

        decisions.append({
            "interval_start_sec": start,
            "assessed_at_sec": end,
            "state": state,
            "bed_status": bed_status,
            "confirmed_absence_sec": absence_seconds,
            "decision": decision,
            "reason": reason,
        })

    return decisions


def main():
    timeline_path = Path("outputs/synthetic_timeline.json")
    events_path = Path("outputs/synthetic_bed_events.json")

    if not timeline_path.exists() or not events_path.exists():
        raise FileNotFoundError(
            "Run activity_timeline.py and bed_events.py first."
        )

    with timeline_path.open("r", encoding="utf-8") as file:
        activity_output = json.load(file)

    with events_path.open("r", encoding="utf-8") as file:
        event_output = json.load(file)

    decisions = generate_decisions(
        activity_output["timeline"],
        event_output["events"],
    )

    print("\nSYNTHETIC TEST — no video was analyzed")
    print(f"Absence alert threshold: {ABSENCE_ALERT_SEC} seconds")
    print("\nMONITORING DECISIONS")

    for result in decisions:
        print(
            f"At {result['assessed_at_sec']:>3}s: "
            f"{result['decision']} — {result['state']}"
        )
        print(f"  Reason: {result['reason']}")

    output = {
        "data_source": "synthetic_test_not_model_predictions",
        "absence_alert_threshold_sec": ABSENCE_ALERT_SEC,
        "decision_timing": "evaluated_at_each_interval_endpoint",
        "unknown_policy": "reset_continuous_absence_timer",
        "decisions": decisions,
    }

    output_path = Path("outputs/synthetic_monitoring.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
