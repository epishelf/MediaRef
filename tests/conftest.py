"""Shared test fixtures for MediaRef test suite."""

import base64
import io
import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import PIL.Image
import pytest

# ============================================================================
# Image Fixtures
# ============================================================================


@pytest.fixture
def sample_image_file(tmp_path: Path) -> Path:
    """Create a sample image file (48x64, solid blue).

    Returns:
        Path to the created PNG image file.
    """
    image_path = tmp_path / "test_image.png"
    test_image = np.zeros((48, 64, 3), dtype=np.uint8)
    test_image[:, :, 2] = 255  # Blue channel (RGB format)
    PIL.Image.fromarray(test_image).save(image_path)
    return image_path


@pytest.fixture
def sample_image_files(tmp_path: Path) -> list[Path]:
    """Create multiple sample image files with different colors.

    Returns:
        List of paths to created image files.
    """
    images = []
    for i in range(3):
        image_path = tmp_path / f"test_image_{i}.png"
        test_image = np.full((48, 64, 3), i * 50, dtype=np.uint8)
        PIL.Image.fromarray(test_image).save(image_path)
        images.append(image_path)
    return images


@pytest.fixture
def sample_rgba_array() -> npt.NDArray[np.uint8]:
    """Create a sample RGBA numpy array with gradient pattern.

    Returns:
        RGBA numpy array (48, 64, 4).
    """
    height, width = 48, 64
    frame = np.zeros((height, width, 4), dtype=np.uint8)

    # Create gradient pattern for easy identification
    for y in range(height):
        for x in range(width):
            frame[y, x] = [x * 4, y * 5, (x + y) * 2, 255]  # RGBA

    return frame


@pytest.fixture
def sample_rgb_array() -> npt.NDArray[np.uint8]:
    """Create a sample RGB numpy array.

    Returns:
        RGB numpy array (48, 64, 3).
    """
    height, width = 48, 64
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Create simple color blocks
    frame[:24, :32] = [255, 0, 0]  # Red
    frame[:24, 32:] = [0, 255, 0]  # Green
    frame[24:, :32] = [0, 0, 255]  # Blue
    frame[24:, 32:] = [255, 255, 0]  # Yellow

    return frame


# ============================================================================
# Video Fixtures
# ============================================================================


def _write_video(path: Path, pixels: np.ndarray, fps: int) -> None:
    height, width = pixels.shape[1:3]
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                str(path),
            ],
            input=pixels.tobytes(),
            check=True,
        )
        return
    # Platforms without the CLI can use the optional fixture encoder.
    av = pytest.importorskip("av", reason="Video fixtures require FFmpeg CLI or PyAV")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.codec_context.thread_count = 1
        for pixels_at_time in pixels:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels_at_time, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def sample_video_file(tmp_path: Path) -> tuple[Path, list[int]]:
    path = tmp_path / "test_video.mp4"
    pixels = np.array([np.full((48, 64, 3), i * 50, dtype=np.uint8) for i in range(5)])
    _write_video(path, pixels, 10)
    return path, [i * 100_000_000 for i in range(5)]


@pytest.fixture
def sample_video_file_large(tmp_path: Path) -> tuple[Path, list[int]]:
    path = tmp_path / "test_video_large.mp4"
    pixels = np.array([np.full((480, 640, 3), (i * 8) % 256, dtype=np.uint8) for i in range(30)])
    _write_video(path, pixels, 30)
    return path, [int(i * 1_000_000_000 / 30) for i in range(30)]


@pytest.fixture
def sample_video_file_long(tmp_path: Path) -> tuple[Path, float]:
    path = tmp_path / "test_video_long.mp4"
    pixels = np.array([np.full((48, 64, 3), (i * 2) % 256, dtype=np.uint8) for i in range(100)])
    _write_video(path, pixels, 10)
    return path, 10.0


# ============================================================================
# URI Fixtures
# ============================================================================


@pytest.fixture
def sample_data_uri(sample_rgba_array: npt.NDArray[np.uint8]) -> str:
    """Create a valid data URI from RGBA array.

    Returns:
        Data URI string (PNG format).
    """
    buffer = io.BytesIO()
    PIL.Image.fromarray(sample_rgba_array).save(buffer, format="PNG")
    base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


@pytest.fixture
def remote_test_image_url() -> str:
    """Return a reliable remote test image URL.

    Returns:
        URL to a test image (httpbingo.org).
    """
    return "https://httpbingo.org/image/png"


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "network: tests requiring network access")
    config.addinivalue_line("markers", "pyav: tests requiring the optional PyAV backend")
    config.addinivalue_line(
        "markers", "tensorcodec: tests requiring the default TensorCodec video backend (tensorcodec-av)"
    )
    config.addinivalue_line("markers", "slow: slow tests (batch processing, large files)")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "performance: performance benchmark tests")


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on markers and available dependencies."""
    # Check the optional PyAV backend.
    try:
        import av  # noqa: F401

        pyav_available = True
    except ImportError:
        pyav_available = False

    skip_pyav = pytest.mark.skip(reason="Optional PyAV backend not installed")

    tensorcodec_available = importlib.util.find_spec("tensorcodec_av") is not None
    skip_tensorcodec = pytest.mark.skip(reason="TensorCodec video backend (tensorcodec-av) not installed")

    for item in items:
        if "tensorcodec" in item.keywords and not tensorcodec_available:
            item.add_marker(skip_tensorcodec)
        # Skip only tests that explicitly require PyAV.
        if "pyav" in item.keywords and not pyav_available:
            item.add_marker(skip_pyav)


def _is_connection_error(exc: BaseException) -> bool:
    try:
        from aiohttp import ClientConnectionError
    except ImportError:
        ClientConnectionError = ConnectionError  # type: ignore[misc,assignment]
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (ConnectionError, TimeoutError, ClientConnectionError)):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Skip, rather than fail, network tests when the remote host is unreachable."""
    outcome = yield
    report = outcome.get_result()
    if (
        report.when == "call"
        and report.failed
        and "network" in item.keywords
        and call.excinfo is not None
        and _is_connection_error(call.excinfo.value)
    ):
        report.outcome = "skipped"
        report.longrepr = (
            str(item.path),
            item.location[1] or 0,
            f"Skipped: network unavailable: {call.excinfo.value}",
        )
