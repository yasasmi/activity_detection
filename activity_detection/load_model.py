import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_model():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU acceleration is unavailable.")

    print(f"Model: {MODEL_ID}", flush=True)
    print("Loading the image/text processor...", flush=True)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=64 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )

    print(
        "Downloading or loading cached model weights...",
        flush=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="mps",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        use_safetensors=True,
    )

    model.eval()

    return model, processor


def main():
    model, processor = load_model()

    first_parameter = next(model.parameters())

    print("\nMODEL LOADING CHECK")
    print("Model class:", type(model).__name__)
    print("Processor class:", type(processor).__name__)
    print("Parameter device:", first_parameter.device)
    print("Parameter data type:", first_parameter.dtype)
    print("Training mode:", model.training)

    print("\nModel loaded successfully.")
    print("No image or video has been analyzed yet.")


if __name__ == "__main__":
    main()
