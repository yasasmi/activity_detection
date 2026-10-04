import argparse
import json
import math
from pathlib import Path


EVENT_TYPES = ("bed_exit", "return_to_bed")


def evaluate_event_type(predictions, annotations, event_type, tolerance):
    predicted = sorted(
        item["start_sec"]
        for item in predictions
        if item["event"] == event_type
    )
    actual = sorted(
        item["start_sec"]
        for item in annotations
        if item["event"] == event_type
    )

    # One-to-one chronological matching within a fixed time tolerance.
    predicted_index = 0
    actual_index = 0
    matches = []

    while predicted_index < len(predicted) and actual_index < len(actual):
        predicted_time = predicted[predicted_index]
        actual_time = actual[actual_index]

        if predicted_time < actual_time - tolerance:
            predicted_index += 1
        elif actual_time < predicted_time - tolerance:
            actual_index += 1
        else:
            matches.append({
                "predicted_start_sec": predicted_time,
                "actual_start_sec": actual_time,
                "absolute_start_error_sec": abs(
                    predicted_time - actual_time
                ),
            })
            predicted_index += 1
            actual_index += 1

    true_positive = len(matches)
    false_positive = len(predicted) - true_positive
    false_negative = len(actual) - true_positive

    precision = (
        true_positive / len(predicted)
        if predicted else None
    )
    recall = (
        true_positive / len(actual)
        if actual else None
    )

    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else None

    return {
        "actual_count": len(actual),
        "predicted_count": len(predicted),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def validate_events(events, duration):
    for item in events:
        if item["event"] not in EVENT_TYPES:
            raise ValueError(f"Unknown event type: {item['event']}")

        timestamp = item["start_sec"]
        if not math.isfinite(timestamp) or not 0 <= timestamp < duration:
            raise ValueError("Invalid event start time.")


def display_metric(value):
    return "N/A (undefined denominator)" if value is None else f"{value:.2%}"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bed events against manual annotations."
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--tolerance-sec", type=float, default=1.0)
    args = parser.parse_args()

    if not math.isfinite(args.tolerance_sec) or args.tolerance_sec < 0:
        raise ValueError("Tolerance must be finite and nonnegative.")

    event_path = Path(args.events)
    truth_path = Path(args.ground_truth)

    predictions = json.loads(event_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))

    if (
        Path(predictions["source_video"]).resolve()
        != Path(truth["video_file"]).resolve()
    ):
        raise ValueError("Events and annotations refer to different videos.")

    duration = truth["duration_sec"]
    predicted_duration = predictions["observation_duration_sec"]

    if (
        not math.isfinite(duration)
        or duration <= 0
        or not math.isfinite(predicted_duration)
        or abs(predicted_duration - duration) > 1e-6
    ):
        raise ValueError("Invalid or mismatched video durations.")

    validate_events(predictions["events"], duration)
    validate_events(truth["events"], duration)

    results = {
        event_type: evaluate_event_type(
            predictions["events"],
            truth["events"],
            event_type,
            args.tolerance_sec,
        )
        for event_type in EVENT_TYPES
    }

    output = {
        "source_video": predictions["source_video"],
        "ground_truth_file": str(truth_path),
        "matching_policy": (
            "Same event type, one-to-one chronological matching "
            "using start_sec within the configured tolerance."
        ),
        "start_time_tolerance_sec": args.tolerance_sec,
        "undefined_metric_policy": "JSON null; displayed as N/A",
        "results": results,
    }

    output_path = event_path.parent / "event_evaluation.json"
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    for event_type, result in results.items():
        print(f"\n{event_type.upper()}")
        print(f"Actual events: {result['actual_count']}")
        print(f"Predicted events: {result['predicted_count']}")
        print(f"True positives: {result['true_positives']}")
        print(f"False positives: {result['false_positives']}")
        print(f"False negatives: {result['false_negatives']}")
        print("Precision:", display_metric(result["precision"]))
        print("Recall:", display_metric(result["recall"]))
        print("F1:", display_metric(result["f1"]))

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
