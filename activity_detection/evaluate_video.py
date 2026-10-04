import argparse
import json
import math
from pathlib import Path

from activity_detection.activity_timeline import (
    VALID_STATES,
    build_timeline,
    calculate_durations,
)
from activity_detection.summarize_bed import summarize_occupancy


BED_STATUSES = ("IN_BED", "OUT_OF_BED", "UNKNOWN")


def validate_ground_truth(timeline, duration):
    if not timeline:
        raise ValueError("Ground-truth timeline is empty.")

    previous_end = 0.0

    for interval in timeline:
        start = interval["start_sec"]
        end = interval["end_sec"]

        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Annotation times must be finite.")

        if abs(start - previous_end) > 1e-6:
            raise ValueError("Ground truth contains a gap or overlap.")

        if end <= start or start < 0 or end > duration:
            raise ValueError("Invalid ground-truth interval.")

        if interval["state"] not in VALID_STATES:
            raise ValueError("Invalid ground-truth activity.")

        if interval["bed_status"] not in BED_STATUSES:
            raise ValueError("Invalid ground-truth bed status.")

        previous_end = end

    if abs(previous_end - duration) > 1e-6:
        raise ValueError("Ground truth must cover the analyzed duration.")


def duration_comparison(labels, predicted, actual):
    return {
        label: {
            "predicted_sec": predicted[label],
            "ground_truth_sec": actual[label],
            "signed_error_sec": predicted[label] - actual[label],
            "absolute_error_sec": abs(predicted[label] - actual[label]),
        }
        for label in labels
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate video predictions against manual annotations."
    )
    parser.add_argument(
        "--predictions",
        default="outputs/video_review_01/observations.json",
    )
    parser.add_argument(
        "--ground-truth",
        default="data/annotations/room_ground_truth.json",
    )
    args = parser.parse_args()

    prediction_path = Path(args.predictions)
    truth_path = Path(args.ground_truth)

    predictions = json.loads(
        prediction_path.read_text(encoding="utf-8")
    )
    truth = json.loads(truth_path.read_text(encoding="utf-8"))

    if (
        Path(predictions["source_video"]).resolve()
        != Path(truth["video_file"]).resolve()
    ):
        raise ValueError("Predictions and ground truth name different videos.")

    duration = predictions["observation_duration_sec"]

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration must be positive and finite.")

    if (
        not math.isfinite(truth["duration_sec"])
        or abs(duration - truth["duration_sec"]) > 1e-6
    ):
        raise ValueError("Prediction and annotation durations differ.")

    observations = predictions["observations"]
    ground_timeline = truth["timeline"]

    validate_ground_truth(ground_timeline, duration)

    # These functions also validate prediction timestamps and labels.
    predicted_timeline = build_timeline(observations, duration)
    predicted_activity = calculate_durations(predicted_timeline)

    _, predicted_bed, _ = summarize_occupancy(
        observations, duration
    )

    actual_activity = {state: 0.0 for state in VALID_STATES}
    actual_bed = {status: 0.0 for status in BED_STATUSES}

    for interval in ground_timeline:
        seconds = interval["end_sec"] - interval["start_sec"]
        actual_activity[interval["state"]] += seconds
        actual_bed[interval["bed_status"]] += seconds

    confusion = {
        actual: {predicted: 0 for predicted in VALID_STATES}
        for actual in VALID_STATES
    }

    activity_correct = 0
    bed_correct = 0
    interval_index = 0

    for observation in observations:
        timestamp = observation["time_sec"]

        while timestamp >= ground_timeline[interval_index]["end_sec"]:
            interval_index += 1

        expected = ground_timeline[interval_index]
        actual_state = expected["state"]
        predicted_state = observation["state"]

        confusion[actual_state][predicted_state] += 1

        activity_correct += predicted_state == actual_state
        bed_correct += (
            observation["bed_status"] == expected["bed_status"]
        )

    count = len(observations)
    activity_accuracy = activity_correct / count
    bed_accuracy = bed_correct / count

    activity_errors = duration_comparison(
        VALID_STATES, predicted_activity, actual_activity
    )
    bed_errors = duration_comparison(
        BED_STATUSES, predicted_bed, actual_bed
    )

    output = {
        "source_video": predictions["source_video"],
        "ground_truth_file": str(truth_path),
        "annotation_source": truth["annotation_source"],
        "duration_sec": duration,
        "observation_count": count,
        "activity_correct_count": activity_correct,
        "activity_accuracy": activity_accuracy,
        "bed_occupancy_correct_count": bed_correct,
        "bed_occupancy_accuracy": bed_accuracy,
        "accuracy_policy": (
            "Unweighted accuracy at sampled timestamps. "
            "UNKNOWN predictions are included."
        ),
        "confusion_matrix_orientation": (
            "Outer keys are actual labels; inner keys are predicted labels."
        ),
        "activity_confusion_matrix": confusion,
        "activity_duration_comparison": activity_errors,
        "bed_duration_comparison": bed_errors,
        "duration_policy": (
            "Prediction labels are held until the next timestamp; "
            "the final label is held until the video ends."
        ),
        "event_evaluation": "Not calculated by this script.",
    }

    output_path = prediction_path.parent / "evaluation.json"
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\nCLASSIFICATION EVALUATION")
    print(
        f"Activity accuracy: {activity_accuracy:.2%} "
        f"({activity_correct}/{count})"
    )
    print(
        f"Bed occupancy accuracy: {bed_accuracy:.2%} "
        f"({bed_correct}/{count})"
    )

    print("\nCONFUSION MATRIX — NONZERO ENTRIES")
    for actual, row in confusion.items():
        for predicted, value in row.items():
            if value:
                print(
                    f"Actual {actual} -> predicted {predicted}: {value}"
                )

    print("\nACTIVITY DURATION COMPARISON")
    for state, result in activity_errors.items():
        print(
            f"{state}: "
            f"predicted={result['predicted_sec']:.2f}s, "
            f"actual={result['ground_truth_sec']:.2f}s, "
            f"absolute error={result['absolute_error_sec']:.2f}s"
        )

    print("\nBED DURATION COMPARISON")
    for status, result in bed_errors.items():
        print(
            f"{status}: "
            f"predicted={result['predicted_sec']:.2f}s, "
            f"actual={result['ground_truth_sec']:.2f}s, "
            f"absolute error={result['absolute_error_sec']:.2f}s"
        )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
