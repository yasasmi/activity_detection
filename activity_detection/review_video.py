import argparse
import json
from pathlib import Path

from activity_detection.analyze_video import analyze_window
from activity_detection.load_model import load_model


MAX_ATTEMPTS = 2
MAX_FRAMES = 8


def review_reasons(current, previous):
    reasons = []

    if not current.get("validation_passed", False):
        reasons.append("invalid_response")

    if (
        current["state"] == "UNKNOWN"
        or current["bed_status"] == "UNKNOWN"
        or current["evidence_quality"] != "clear"
    ):
        reasons.append("uncertain_observation")

    if current["bed_status"] == "OUT_OF_BED":
        reasons.append("outside_bed_needs_movement_evidence")

    if previous is not None:
        if (
            current["state"] != previous["state"]
            or current["bed_status"] != previous["bed_status"]
        ):
            reasons.append("activity_or_occupancy_transition")

    return reasons


def choose_context(frames, target_index, radius):
    target_time = frames[target_index]["time_sec"]

    indices = [
        index
        for index, frame in enumerate(frames)
        if abs(frame["time_sec"] - target_time) <= radius
    ]

    if len(indices) > MAX_FRAMES:
        # Spread context across the requested range.
        selected = {
            indices[round(k * (len(indices) - 1) / 6)]
            for k in range(7)
        }
        selected.add(target_index)
        indices = sorted(selected)

    return [frames[index] for index in indices]


def valid_movement(observation):
    for name in ("moving_away", "approaching_bed"):
        if name not in observation:
            return False
        value = observation[name]
        if value is not None and type(value) is not bool:
            return False

    if (
        observation["moving_away"] is True
        and observation["approaching_bed"] is True
    ):
        return False

    evidence = observation.get("movement_evidence")
    return isinstance(evidence, str) and bool(evidence.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="outputs/video_run_01/observations.json",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/video_frames/manifest.json",
    )
    parser.add_argument("--out", default="outputs/video_review_01")
    args = parser.parse_args()

    source = json.loads(
        Path(args.predictions).read_text(encoding="utf-8")
    )
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    frames = manifest["frames"]
    original = source["observations"]

    if source["source_video"] != manifest["source_video"]:
        raise ValueError("Predictions and frames refer to different videos.")

    if [x["time_sec"] for x in original] != [
        x["time_sec"] for x in frames
    ]:
        raise ValueError("Prediction and frame timestamps do not match.")

    output_folder = Path(args.out)
    output_folder.mkdir(parents=True, exist_ok=False)

    model = processor = None
    reviewed = []
    review_count = 0
    model_calls = 0

    with (output_folder / "review_log.jsonl").open(
        "w", encoding="utf-8"
    ) as log:
        for index, initial in enumerate(original):
            previous = reviewed[-1] if reviewed else None
            reasons = review_reasons(initial, previous)
            result = dict(initial)
            attempts = []

            # Missing direction evidence is not a negative observation.
            result.update({
                "moving_away": None,
                "approaching_bed": None,
                "movement_evidence": "Direction was not assessed.",
                "available_at_sec": initial["time_sec"],
            })

            if reasons:
                review_count += 1

                if model is None:
                    model, processor = load_model()

                resolved = False

                for attempt in range(1, MAX_ATTEMPTS + 1):
                    window = choose_context(frames, index, radius=2 * attempt)

                    print(
                        f"Review {initial['time_sec']:.2f}s, "
                        f"attempt {attempt}/{MAX_ATTEMPTS}",
                        flush=True,
                    )

                    observation, raw, error = analyze_window(
                        model,
                        processor,
                        window,
                        manifest_path.parent,
                        target_time=initial["time_sec"],
                    )
                    model_calls += 1

                    movement_ok = valid_movement(observation)

                    resolved = (
                        error is None
                        and movement_ok
                        and observation["state"] != "UNKNOWN"
                        and observation["bed_status"] != "UNKNOWN"
                        and observation["evidence_quality"] == "clear"
                    )

                    if observation["bed_status"] == "OUT_OF_BED":
                        resolved = resolved and (
                            observation.get("moving_away") is not None
                            and observation.get("approaching_bed") is not None
                        )

                    attempts.append({
                        "attempt": attempt,
                        "context_timestamps_sec": [
                            frame["time_sec"] for frame in window
                        ],
                        "raw_response": raw,
                        "validation_error": error,
                        "movement_schema_valid": movement_ok,
                        "resolved": resolved,
                    })

                    result["available_at_sec"] = max(
                        frame["time_sec"] for frame in window
                    )

                    if resolved:
                        result.update(observation)
                        result["raw_response"] = raw
                        result["validation_passed"] = True
                        result["validation_error"] = None
                        result["context_timestamps_sec"] = [
                            frame["time_sec"] for frame in window
                        ]
                        break

                if not resolved:
                    result.update({
                        "state": "UNKNOWN",
                        "bed_status": "UNKNOWN",
                        "person_visible": None,
                        "evidence_quality": "insufficient",
                        "evidence": "Context review exhausted without resolution.",
                        "moving_away": None,
                        "approaching_bed": None,
                        "movement_evidence": "Unresolved after bounded review.",
                        "validation_passed": False,
                        "validation_error": "context_review_unresolved",
                        "raw_response": attempts[-1]["raw_response"],
                        "context_timestamps_sec": (
                            attempts[-1]["context_timestamps_sec"]
                        ),
                    })

            result["review_reasons"] = reasons
            result["review_attempts"] = len(attempts)
            reviewed.append(result)

            log.write(json.dumps({
                "time_sec": initial["time_sec"],
                "reasons": reasons,
                "attempts": attempts,
                "result": result,
            }) + "\n")
            log.flush()

    output = dict(source)
    output["observations"] = reviewed
    output["review_policy"] = {
        "max_attempts": MAX_ATTEMPTS,
        "max_context_frames": MAX_FRAMES,
        "context_radius_sec": [2, 4],
        "mode": "offline_review_with_possible_future_context",
        "unresolved_policy": "UNKNOWN",
    }

    output_path = output_folder / "observations.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nCONTEXT REVIEW COMPLETE")
    print(f"Observations: {len(reviewed)}")
    print(f"Observations requesting review: {review_count}")
    print(f"Additional model calls: {model_calls}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
