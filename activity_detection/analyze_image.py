import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from qwen_vl_utils import process_vision_info

from activity_detection.load_model import load_model


def main():
    parser = argparse.ArgumentParser(
        description="Analyze one image using Qwen."
    )
    parser.add_argument("image_path")
    args = parser.parse_args()

    image_path = Path(args.image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Correct image orientation and converting to RGB.
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    model, processor = load_model()

    prompt = (
        "Describe this image in two short sentences. "
        "If a person is visible, describe their visible posture "
        "and position relative to any visible bed. "
        "If no person is visible, say so. "
        "State uncertainty when details are unclear. "
        "Do not infer movement direction, a bed exit, "
        "or a return to bed from this single image."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": 64 * 28 * 28,
                    "max_pixels": 256 * 28 * 28,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

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

    print("\nAnalyzing image...", flush=True)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    # Keeping only the newly generated answer.
    input_length = inputs["input_ids"].shape[1]
    answer_ids = generated_ids[:, input_length:]

    answer = processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    print("\nMODEL RESPONSE")
    print(answer)

    output_path = Path("outputs/image_test.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"Input image: {image_path}\n\n{answer}\n",
        encoding="utf-8",
    )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
