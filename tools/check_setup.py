import sys
import shutil
from pathlib import Path

print("Python version:", sys.version.split()[0])
print("Python environment:", sys.executable)
print("Project folder:", Path.cwd())
print("FFmpeg location:", shutil.which("ffmpeg"))
print("FFprobe location:", shutil.which("ffprobe"))

for folder in ("data", "outputs"):
    Path(folder).mkdir(exist_ok=True)
    print(f"Folder ready: {folder}")
