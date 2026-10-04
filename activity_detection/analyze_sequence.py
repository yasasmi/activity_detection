import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from qwen_vl_utils import process_vision_info

from activity_detection.load_model import load_model
from activity_detection.recognize_activity import validate_observation


PROMPT = """
These images are time-ordered samples from one video.
Each image is preceded by its timestamp in seconds.

Describe the person's activity AT THE FINAL IMAGE.
Use earlier images only as context for visible movement.

Return one JSON object containing these five fields:
- state: LYING_IN_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED,
  STANDING, WALKING, OUT_OF_BED or UNKNOWN.
- bed_status: IN_BED, OUT_OF_BED or UNKNOWN.
- person_visible: a JSON boolean, true or false.
- evidence_quality: clear, partial or insufficient.
- evidence: a short description of visible posture and movement.

Choose all values from the images. No default answer is supplied.

Definitions:
- SITTING_ON_BED: pelvis supported by the bed and torso upright
  or mostly upright. Legs may be on the bed or hanging over
  its edge. Back support and extended legs alone do not mean lying.
- LYING_IN_BED: torso reclined along the mattress, with the
  person lying on their back, side or front.
- SITTING_OUTSIDE_BED: sitting on a surface outside the bed.
- STANDING: upright on their feet on the floor.
- WALKING: visible stepping and movement across frames.
- OUT_OF_BED: clearly outside the bed, specific activity unclear.
- UNKNOWN: insufficient evidence to determine activity.

person_visible describes visibility in the final image:
- Use true if a person is visible, even if posture is unclear.
- Use false only if no person is visible.
- If multiple people make the target ambiguous, activity and
  occupancy should be UNKNOWN, but visibility can still be true.

Use IN_BED for sitting or lying supported by the bed.
Use OUT_OF_BED when clearly outside the bed.
Use UNKNOWN when the relationship to the bed is unclear.
If no person is visible in the final image, activity and
bed_status must be UNKNOWN.

Do not infer walking from a single upright pose.
Hand movement or moving legs while sitting does not establish walking.
Do not infer person movement from camera motion or across scene cuts.
Do not infer bed-exit or return events in this step.
Do not guess hidden posture.

The activity label must agree with the evidence description.
Return only JSON, without markdown.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Analyze four consecutive video frames."
    )
    parser.add_argument(
        "--manifest",
        default="outputs/video_frames/manifest.json",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    frames = [
        frame
        for frame in manifest["frames"]
        if frame["time_sec"] >= args.start_sec
    ][:4]

    if len(frames) < 4:
        raise ValueError(
            "This test needs four frames. Choose an earlier start time."
        )

    content = [{"type": "text", "text": PROMPT}]

    for frame in frames:
        image_path = manifest_path.parent / frame["file"]

        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        content.append({
            "type": "text",
            "text": f"Timestamp: {frame['time_sec']:.3f} seconds",
        })
        content.append({
            "type": "image",
            "image": image,
            "min_pixels": 64 * 28 * 28,
            "max_pixels": 256 * 28 * 28,
        })

    messages = [{"role": "user", "content": content}]

    print("Frame timestamps:", [
        frame["time_sec"] for frame in frames
    ], flush=True)

    model, processor = load_model()

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

    print("\nAnalyzing the frame sequence...", flush=True)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]

    raw_response = processor.batch_decode(
        generated_ids[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    output_folder = Path("outputs")
    output_folder.mkdir(exist_ok=True)

    tag = f"{frames[0]['time_sec']:.3f}".replace(".", "_")

    raw_path = output_folder / f"sequence_{tag}_raw.txt"
    raw_path.write_text(raw_response, encoding="utf-8")

    json_text = raw_response
    if json_text.startswith("```") and json_text.endswith("```"):
        json_text = "\n".join(json_text.splitlines()[1:-1])

    validation_error = None

    try:
        observation = validate_observation(
            json.loads(json_text),
            allow_walking=True,
        )
    except (ValueError, TypeError) as error:
        validation_error = str(error)
        observation = None

    output = {
        "data_source": "video_frame_sequence_model_prediction",
        "source_video": manifest["source_video"],
        "context_timestamps_sec": [
            frame["time_sec"] for frame in frames
        ],
        "observation_time_sec": frames[-1]["time_sec"],
        "validation_passed": validation_error is None,
        "validation_error": validation_error,
        "observation": observation,
        "raw_response_file": str(raw_path),
    }

    output_path = output_folder / f"sequence_{tag}.json"
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    if validation_error:
        print("\nVALIDATION FAILED:", validation_error)
        print(raw_response)
    else:
        print("\nVALIDATED SEQUENCE OBSERVATION")
        print(json.dumps(observation, indent=2))

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
