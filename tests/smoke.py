"""Release smoke test — runs against the built wheel/sdist in isolation.

Catches packaging-time breakage that the in-repo pytest suite cannot see:
missing files in the dist, broken ``__version__``, missing entry points,
core API regressions when only the core deps are available.

Invoked from ``.github/workflows/publish.yml`` after ``uv build``, before
publish, with ``uv run --isolated --no-project --with dist/*.whl`` —
so only stdlib + core deps are importable here.
"""

import importlib.metadata
import json
from pathlib import Path

import mediaref
from mediaref import DataURI, MediaRef, batch_decode, cleanup_cache  # noqa: F401

# Real version from hatch-vcs (git tag), not the dev fallback.
assert mediaref.__version__ and mediaref.__version__ != "0.0.0.dev0", f"unexpected version: {mediaref.__version__!r}"

# Public __all__ symbols all resolve.
for _name in mediaref.__all__:
    getattr(mediaref, _name)

# py.typed + _version.py shipped in the dist.
pkg = Path(mediaref.__file__).parent
assert (pkg / "py.typed").is_file(), "py.typed missing"
assert (pkg / "_version.py").is_file(), "_version.py missing"

# Wire-format round-trip without any optional decoder.
ref = MediaRef(uri="video.mp4", pts_ns=1_500_000_000)
j = ref.model_dump_json()
assert MediaRef.model_validate_json(j).model_dump() == ref.model_dump()
assert json.loads(j) == {"uri": "video.mp4", "pts_ns": 1_500_000_000}
assert MediaRef(uri="s3://b/x.mp4", pts_ns=0).is_cloud_uri is True
assert MediaRef(uri="x.png").is_cloud_uri is False

# DataURI parsing and decoding (proves data_uri + TensorCodec/OpenCV image decoding shipped).
_pixel = DataURI.from_uri(
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVQIHWNgAAAAAgABz8g15QAAAABJRU5ErkJggg=="
).to_ndarray(format="rgba")
assert _pixel.shape == (1, 1, 4), _pixel.shape

# cleanup_cache must be safe without the [video] extra.
cleanup_cache()

# CLI entry point is installed.
# compatibility — entry_points(group=...) is py3.10+.
eps = importlib.metadata.distribution("mediaref").entry_points
assert any(ep.group == "console_scripts" and ep.name == "mediaref" for ep in eps), "mediaref CLI entry point missing"

print(f"smoke OK — mediaref {mediaref.__version__}")
