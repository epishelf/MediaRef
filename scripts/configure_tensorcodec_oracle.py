"""Use TensorCodec's FFmpeg in Linux CI's optional TorchCodec oracle."""

import importlib.util
import os
import re
from pathlib import Path

spec = importlib.util.find_spec("tensorcodec_av")  # FFmpeg ships in tensorcodec-av since 0.3
libs = Path(spec.origin).parent.parent / "tensorcodec_av.libs"
if not libs.is_dir():
    raise RuntimeError("Expected an installed tensorcodec-av wheel with bundled FFmpeg")
for library in libs.glob("lib*.so.*"):
    match = re.fullmatch(r"(lib[^-]+)-[0-9a-f]+(\.so\.\d+)", library.name)
    if match and match[1] in {"libavcodec", "libavformat", "libavutil", "libswscale", "libswresample"}:
        alias = libs / (match[1] + match[2])
        if not alias.exists():
            alias.symlink_to(library.name)
with open(os.environ["GITHUB_ENV"], "a") as output:
    output.write(f"LD_LIBRARY_PATH={libs}\n")
