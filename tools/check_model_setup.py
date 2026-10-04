import platform
import sys

import torch
import torchvision
import transformers

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


print("Python:", sys.version.split()[0])
print("Architecture:", platform.machine())
print("macOS:", platform.mac_ver()[0])

print("\nLIBRARY VERSIONS")
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Transformers:", transformers.__version__)

print("\nAPPLE GPU CHECK")
print("MPS built into PyTorch:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    # Perform a small calculation on the Apple GPU.
    numbers = torch.tensor([1.0, 2.0, 3.0], device="mps")
    result = (numbers * 2).cpu().tolist()

    print("GPU calculation result:", result)
    print("Apple GPU check passed.")
else:
    print("Apple GPU acceleration is not currently available.")

print("\nQwen model imports passed.")
print("No model weights have been downloaded or loaded yet.")