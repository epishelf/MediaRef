"""Tests for mediaref's import-time behavior.

These run in subprocesses because they assert on a fresh interpreter's
``sys.modules`` state — mutating the caller's module table would break any
other test that already holds bound references to ``mediaref`` symbols.
"""

import subprocess
import sys
import textwrap

import pytest

from tests import _torchcodec_video_available


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )


@pytest.mark.pyav
def test_pyav_only_import_does_not_load_torchcodec():
    """Loading the PyAV backend must not transitively load TorchCodec."""
    result = _run("""
        import sys
        from mediaref.video_decoder import PyAVVideoDecoder  # noqa: F401
        leaked = [m for m in sys.modules if m.startswith("torchcodec")]
        assert not leaked, f"torchcodec leaked: {leaked}"
        print("OK")
    """)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "OK"


def test_common_video_decoder_import_loads_neither_backend():
    result = _run("""
        import sys
        import mediaref.video_decoder  # noqa: F401
        assert "av" not in sys.modules
        assert not [m for m in sys.modules if m.startswith("torchcodec")]
        print("OK")
    """)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "OK"


def test_torchcodec_backend_does_not_require_pyav():
    result = _run("""
        import sys
        from types import ModuleType

        torchcodec = ModuleType("torchcodec")
        decoders = ModuleType("torchcodec.decoders")
        decoders.VideoDecoder = type("VideoDecoder", (), {})
        torchcodec.decoders = decoders
        sys.modules["torchcodec"] = torchcodec
        sys.modules["torchcodec.decoders"] = decoders
        sys.modules["av"] = None

        from mediaref.video_decoder import TorchCodecVideoDecoder  # noqa: F401

        assert "mediaref.video_decoder.pyav_decoder" not in sys.modules
        print("OK")
    """)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "OK"


def test_pyav_backend_has_specific_install_error():
    result = _run("""
        import sys
        sys.modules["av"] = None

        import mediaref.video_decoder  # common interfaces remain available
        try:
            from mediaref.video_decoder import PyAVVideoDecoder  # noqa: F401
        except ImportError as error:
            assert "mediaref[pyav]" in str(error)
            assert "mediaref[video]" not in str(error)
        else:
            raise AssertionError("PyAV import unexpectedly succeeded")
        print("OK")
    """)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "OK"


def test_top_level_mediaref_import_does_not_load_torchcodec():
    """`import mediaref` must not load torchcodec either. Runs unconditionally
    — the top-level package has no PyAV dependency."""
    result = _run("""
        import sys
        import mediaref  # noqa: F401
        leaked = [m for m in sys.modules if m.startswith("torchcodec")]
        assert not leaked, f"torchcodec leaked: {leaked}"
        print("OK")
    """)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "OK"


def test_torchcodec_video_availability_calls_ffmpeg_operation():
    calls = 0

    def get_ffmpeg_library_versions():
        nonlocal calls
        calls += 1

    assert _torchcodec_video_available(get_ffmpeg_library_versions)
    assert calls == 1


def test_torchcodec_video_availability_rejects_missing_ffmpeg():
    def get_ffmpeg_library_versions():
        raise RuntimeError("FFmpeg is unavailable")

    assert not _torchcodec_video_available(get_ffmpeg_library_versions)


def test_missing_tensorcodec_av_keeps_images_and_names_video_alternatives():
    """Without tensorcodec-av (e.g. Windows), images decode and default video raises a clear ImportError."""
    result = _run("""
        import sys
        sys.modules["tensorcodec_av"] = None

        import numpy as np
        from mediaref import DataURI, MediaRef, batch_decode

        rgb = np.zeros((2, 3, 3), np.uint8)
        assert MediaRef(uri=DataURI.from_image(rgb).uri).to_ndarray().shape == (2, 3, 3)

        ref = MediaRef(uri="video.mp4", pts_ns=0)
        for call in (ref.to_ndarray, lambda: batch_decode([ref])):
            try:
                call()
            except ImportError as error:
                assert "tensorcodec-av" in str(error), error
                assert "mediaref[pyav]" in str(error), error
            else:
                raise AssertionError("default video decoding unexpectedly succeeded")
        print("OK")
    """)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "OK"
