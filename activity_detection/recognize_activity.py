import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from qwen_vl_utils import process_vision_info

from activity_detection.load_model import load_model
from activity_detection.activity_timeline import VALID_STATES


PROMPT = """
Inspect the image and classify the visible person's posture.

Return one JSON object with exactly these five fields:
- state: one of LYING_IN_BED, SITTING_ON_BED,
  SITTING_OUTSIDE_BED, STANDING, OUT_OF_BED, UNKNOWN.
- bed_status: one of IN_BED, OUT_OF_BED, UNKNOWN.
- person_visible: a JSON boolean, true or false.
- evidence_quality: one of clear, partial, insufficient.
- evidence: a short description of the visible posture.

Choose every field from the image. No default answer is supplied.

Definitions:
SITTING_ON_BED means the pelvis is supported by the bed and
the torso is upright or mostly upright. Legs may be on the
bed or hanging over its edge.

LYING_IN_BED means the torso is reclined along the mattress,
with the person lying on their back, side or front.

SITTING_OUTSIDE_BED means sitting on a chair or another
surface outside the bed.

STANDING means upright on their feet on the floor.

OUT_OF_BED means clearly outside the bed, but the specific
posture is unclear.

UNKNOWN means the activity cannot be determined.

Visibility and certainty are separate:
- person_visible is true when a person is visibly present,
  even if their posture is unclear or partially obscured.
- person_visible is false only when no person is visible.
- If multiple people make the target ambiguous, use UNKNOWN
  for activity and occupancy; visibility can still be true.

Use IN_BED for sitting or lying supported by the bed.
Use OUT_OF_BED when the person is clearly outside the bed.
Use UNKNOWN when their relationship to the bed is unclear.

Do not infer walking, a bed exit or a return from one image.
Check that the fields agree with the evidence description.
Return only JSON, without markdown.
"""


def validate_observation(observation, allow_walking=False):
    """Check format and reject contradictory observations."""

    if not isinstance(observation, dict):
        raise ValueError("Expected a JSON object.")

    state = observation.get("state")
    bed_status = observation.get("bed_status")

    if state not in VALID_STATES:
        raise ValueError(f"Invalid state: {state}")

    if state == "WALKING" and not allow_walking:
        raise ValueError("Walking cannot be confirmed in this still-image test.")

    if bed_status not in ("IN_BED", "OUT_OF_BED", "UNKNOWN"):
        raise ValueError("Invalid bed_status.")

    if type(observation.get("person_visible")) is not bool:
        raise ValueError("person_visible must be true or false.")

    quality = observation.get("evidence_quality")

    if quality not in ("clear", "partial", "insufficient"):
        raise ValueError("Invalid evidence_quality.")

    evidence = observation.get("evidence")

    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("A short evidence description is required.")

    if not observation["person_visible"] or quality == "insufficient":
        observation["state"] = "UNKNOWN"
        observation["bed_status"] = "UNKNOWN"
        return observation

    expected_occupancy = {
        "LYING_IN_BED": "IN_BED",
        "SITTING_ON_BED": "IN_BED",
        "SITTING_OUTSIDE_BED": "OUT_OF_BED",
        "STANDING": "OUT_OF_BED",
        "WALKING": "OUT_OF_BED",
        "OUT_OF_BED": "OUT_OF_BED",
    }

    if (
        state in expected_occupancy
        and bed_status != expected_occupancy[state]
    ):
        raise ValueError("Activity and bed occupancy contradict each other.")

    return observation


def main():
    parser = argparse.ArgumentParser(
        description="Recognize visible activity in one image."
    )
    parser.add_argument("image_path")
    args = parser.parse_args()

    image_path = Path(args.image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    model, processor = load_model()

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
                "min_pixels": 64 * 28 * 28,
                "max_pixels": 256 * 28 * 28,
            },
            {"type": "text", "text": PROMPT},
        ],
    }]

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

    print("\nRecognizing activity...", flush=True)

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

    # Preserve the original response for debugging.
    output_folder = Path("outputs")
    output_folder.mkdir(exist_ok=True)

    (output_folder / "activity_raw_response.txt").write_text(
        raw_response,
        encoding="utf-8",
    )

    # Accept a surrounding markdown fence if the model adds one.
    json_text = raw_response
    if json_text.startswith("```") and json_text.endswith("```"):
        json_text = "\n".join(json_text.splitlines()[1:-1])

    try:
        observation = json.loads(json_text)
        observation = validate_observation(observation)
    except (ValueError, TypeError) as error:
        print("\nResponse could not be validated:", error)
        print("\nRAW MODEL RESPONSE")
        print(raw_response)
        return

    print("\nVALIDATED ACTIVITY OBSERVATION")
    print(json.dumps(observation, indent=2))

    output = {
        "data_source": "single_image_model_prediction",
        "image_path": str(image_path),
        "observation": observation,
    }

    output_path = output_folder / "image_activity.json"
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
