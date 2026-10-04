import json
from pathlib import Path


IN_BED_STATES = {"LYING_IN_BED", "SITTING_ON_BED"}

OUTSIDE_STATES = {
    "STANDING",
    "WALKING",
    "SITTING_OUTSIDE_BED",
    "OUT_OF_BED",
}


def detect_bed_events(timeline, movement_evidence):
    """Detect events using activity history and movement evidence."""

    events = []

    bed_status = "UNKNOWN"
    last_in_bed_state = None

    pending_exit = None
    pending_return = None
    approach_seen = False

    for interval in timeline:
        start = interval["start_sec"]
        end = interval["end_sec"]
        state = interval["state"]

        movement = movement_evidence.get(start, {})
        moving_away = movement.get("moving_away", False)
        approaching = movement.get("approaching_bed", False)

        # Missing evidence breaks the event sequence.
        if state == "UNKNOWN":
            bed_status = "UNKNOWN"
            last_in_bed_state = None
            pending_exit = None
            pending_return = None
            approach_seen = False
            continue

        if state in OUTSIDE_STATES:
            # The person moved outside after being observed in bed.
            if bed_status == "IN_BED":
                pending_exit = {
                    "start_sec": start,
                    "previous_state": last_in_bed_state,
                }

            bed_status = "OUT_OF_BED"

            # Leaving again cancels an incomplete return sequence.
            pending_return = None

            # Walking away confirms a pending departure.
            if pending_exit and state == "WALKING" and moving_away:
                events.append({
                    "event": "bed_exit",
                    "start_sec": pending_exit["start_sec"],
                    "confirmed_sec": end,
                    "previous_state": pending_exit["previous_state"],
                    "current_state": state,
                    "evidence": (
                        "Previously in bed, then outside bed "
                        "and observed walking away."
                    ),
                })

                # Clear the candidate so later walking is not
                # counted as another exit.
                pending_exit = None

            approach_seen = approach_seen or approaching
            continue

        if state in IN_BED_STATES:
            # Sitting back down cancels an unconfirmed exit.
            pending_exit = None

            if state == "SITTING_ON_BED":
                if bed_status == "OUT_OF_BED" and approach_seen:
                    pending_return = {
                        "start_sec": start,
                        "previous_state": "OUT_OF_BED",
                    }

            if state == "LYING_IN_BED" and pending_return:
                events.append({
                    "event": "return_to_bed",
                    "start_sec": pending_return["start_sec"],
                    "confirmed_sec": end,
                    "previous_state": pending_return["previous_state"],
                    "current_state": state,
                    "evidence": (
                        "Previously outside bed, approached, "
                        "sat on the bed, then lay down."
                    ),
                })

                pending_return = None

            bed_status = "IN_BED"
            last_in_bed_state = state
            approach_seen = False

    return events


def main():
    input_path = Path("outputs/synthetic_timeline.json")

    if not input_path.exists():
        raise FileNotFoundError(
            "Run python activity_timeline.py first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        activity_output = json.load(file)

    # synthetic examples
    # identifying the start time of the relevant interval.
    
    movement_evidence = {
        20: {"moving_away": True},
        45: {"approaching_bed": True},
    }

    events = detect_bed_events(
        activity_output["timeline"],
        movement_evidence,
    )

    exit_count = sum(
        event["event"] == "bed_exit" for event in events
    )

    return_count = sum(
        event["event"] == "return_to_bed" for event in events
    )

    print("\nSYNTHETIC TEST — no video was analyzed")
    print("\nBED EVENTS")

    for event in events:
        print(
            f"{event['event'].upper()}: "
            f"started at {event['start_sec']}s, "
            f"confirmed by {event['confirmed_sec']}s"
        )
        print(f"  Evidence: {event['evidence']}")

    print(f"\nBed exits: {exit_count}")
    print(f"Bed returns: {return_count}")

    output = {
        "data_source": "synthetic_test_not_model_predictions",
        "confirmation_policy": "end_of_supporting_interval",
        "bed_exit_count": exit_count,
        "bed_return_count": return_count,
        "events": events,
    }

    output_path = Path("outputs/synthetic_bed_events.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
