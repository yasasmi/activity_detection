import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from qwen_vl_utils import process_vision_info

from activity_detection.analyze_sequence import PROMPT
from activity_detection.load_model import load_model
from activity_detection.recognize_activity import validate_observation


def analyze_window(
    model, processor, window, frame_folder, target_time=None
):
    instruction = PROMPT

    if target_time is not None:
        instruction = instruction.replace(
            "Describe the person's activity AT THE FINAL IMAGE.",
            (
                f"Describe the person's activity AT TIMESTAMP "
                f"{target_time:.3f} seconds."
            ),
        )

        instruction = instruction.replace(
            "Use earlier images only as context for visible movement.",
            "Use the surrounding images as context for visible movement.",
        )

        instruction = instruction.replace(
            "final image",
            f"target image at {target_time:.3f} seconds",
        )

        instruction += """
For this review, use the complete eight-field JSON format below
instead of the earlier five-field example. Include every field.

{
  "state": "UNKNOWN",
  "bed_status": "UNKNOWN",
  "person_visible": false,
  "evidence_quality": "insufficient",
  "evidence": "Describe the target person's visible posture",
  "moving_away": null,
  "approaching_bed": null,
  "movement_evidence": "Describe movement relative to the bed"
}

These example values illustrate the format, not the answer.

Distinguish posture using visible evidence:
- SITTING_ON_BED: pelvis supported by the bed and torso upright
  or mostly upright. Back support and extended legs alone do
  not establish lying.
- LYING_IN_BED: torso reclined along the mattress, with the
  person lying on their back, side or front.
- UNKNOWN: posture cannot be determined.

The activity label must agree with the evidence description.

moving_away and approaching_bed must each be true, false or null.
Use null when direction relative to the bed cannot be established.
Use true only for movement visible leading up to and including
the target timestamp.
Do not attribute movement after the target timestamp to it.
Do not infer movement from camera motion or a scene cut.
Do not mark both directions true.

Return only one JSON object containing all eight fields.
"""

    if len(window) == 1:
        instruction += """
Only one image is available.
Do not select WALKING because there is no movement context.
"""

    content = [{"type": "text", "text": instruction}]

    for frame in window:
        image_path = frame_folder / frame["file"]

        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        content.extend([
            {
                "type": "text",
                "text": (
                    f"Timestamp: {frame['time_sec']:.3f} seconds"
                ),
            },
            {
                "type": "image",
                "image": image,
                "min_pixels": 64 * 28 * 28,
                "max_pixels": 256 * 28 * 28,
            },
        ])

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("mps")

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]

    raw = processor.batch_decode(
        generated[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    json_text = raw

    if json_text.startswith("```") and json_text.endswith("```"):
        json_text = "\n".join(json_text.splitlines()[1:-1])

    validation_error = None

    try:
        observation = validate_observation(
            json.loads(json_text),
            allow_walking=len(window) > 1,
        )
    except (ValueError, TypeError) as error:
        validation_error = str(error)

        observation = {
            "state": "UNKNOWN",
            "bed_status": "UNKNOWN",
            "person_visible": None,
            "evidence_quality": "insufficient",
            "evidence": (
                "Model response failed validation; "
                "visibility and activity were not established."
            ),
        }

    return observation, raw, validation_error


def main():
    parser = argparse.ArgumentParser(
        description="Analyze all sampled video frames."
    )

    parser.add_argument(
        "--manifest",
        default="outputs/video_frames/manifest.json",
    )

    parser.add_argument(
        "--out",
        default="outputs/video_run_01",
    )

    args = parser.parse_args()
    manifest_path = Path(args.manifest)

    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    frames = manifest["frames"]

    if not frames:
        raise ValueError("The manifest contains no frames.")

    # Check images before loading the model.
    for frame in frames:
        image_path = manifest_path.parent / frame["file"]

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

    output_folder = Path(args.out)

    if output_folder.exists():
        raise FileExistsError(
            f"{output_folder} already exists. "
            "Choose a new output folder with --out."
        )

    output_folder.mkdir(parents=True)

    model, processor = load_model()
    observations = []

    checkpoint_path = output_folder / "observations.jsonl"

    with checkpoint_path.open("w", encoding="utf-8") as checkpoint:
        for index, frame in enumerate(frames):
            # Current frame plus up to three preceding frames.
            window = frames[max(0, index - 3):index + 1]

            print(
                f"\n[{index + 1}/{len(frames)}] "
                f"Analyzing {frame['time_sec']:.2f}s...",
                flush=True,
            )

            observation, raw, error = analyze_window(
                model,
                processor,
                window,
                manifest_path.parent,
            )

            record = {
                "time_sec": frame["time_sec"],
                **observation,
                "context_timestamps_sec": [
                    item["time_sec"] for item in window
                ],
                "validation_passed": error is None,
                "validation_error": error,
                "raw_response": raw,
            }

            observations.append(record)

            # immediately Saving each completed observation.
            checkpoint.write(json.dumps(record) + "\n")
            checkpoint.flush()

            print(
                f"  {record['state']} | {record['bed_status']}",
                flush=True,
            )

            if error:
                print(
                    f"  Validation error: {error}",
                    flush=True,
                )

    output = {
        "data_source": "video_model_predictions",
        "source_video": manifest["source_video"],
        "observation_duration_sec": manifest["analyzed_duration_sec"],
        "sample_fps": manifest["sample_fps"],
        "context_policy": "current frame and up to 3 earlier frames",
        "observations": observations,
    }

    output_path = output_folder / "observations.json"

    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    invalid_count = sum(
        not record["validation_passed"]
        for record in observations
    )

    print("\nVIDEO ANALYSIS COMPLETE")
    print(f"Observations: {len(observations)}")
    print(f"Invalid responses: {invalid_count}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()