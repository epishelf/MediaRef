"""Tests for fsspec-backed cloud-storage URIs.

Uses fsspec's built-in ``memory://`` filesystem so the suite runs fully
offline — no S3, no GCS, no network.
"""

from __future__ import annotations

import io

import fsspec
import numpy as np
import numpy.typing as npt
import PIL.Image
import pytest

from mediaref import MediaRef, batch_decode, cleanup_cache
from mediaref._internal import _DIRECT_URI_SCHEMES, is_cloud_uri

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_memory_fs():
    """Wipe the in-process MemoryFileSystem AND mediaref's container cache
    between tests.

    The container cache is keyed by URI string. Without clearing it, a
    second test that re-creates ``memory://x`` with different bytes would
    HIT the previous test's cached container (still holding the original
    bytes via its owned MemoryFile) — silent cross-test pollution.
    """
    cleanup_cache()
    fs = fsspec.filesystem("memory")
    # Reset the underlying store directly (no public clear()).
    if hasattr(fs, "store"):
        fs.store.clear()
    if hasattr(fs, "pseudo_dirs"):
        fs.pseudo_dirs.clear()
    yield
    cleanup_cache()
    if hasattr(fs, "store"):
        fs.store.clear()
    if hasattr(fs, "pseudo_dirs"):
        fs.pseudo_dirs.clear()


def _put_bytes(uri: str, data: bytes) -> None:
    """Write raw bytes to an fsspec URI."""
    with fsspec.open(uri, "wb") as f:
        f.write(data)


def _png_bytes(rgb_image: npt.NDArray[np.uint8]) -> bytes:
    """Encode an RGB array to PNG bytes via Pillow."""
    buffer = io.BytesIO()
    PIL.Image.fromarray(rgb_image).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# is_cloud_uri / properties
# ---------------------------------------------------------------------------


class TestIsCloudUri:
    @pytest.mark.parametrize(
        "uri",
        [
            "http://example.com/x.png",
            "https://example.com/x.png",
            "s3://bucket/key.mp4",
            "gs://bucket/key.png",
            "hf://datasets/foo/bar.mp4",
            "az://container/blob.png",
            "abfs://container/blob.png",
            "ftp://host/path.png",
            "memory://test/file.png",
        ],
    )
    def test_recognized_cloud_schemes(self, uri):
        assert is_cloud_uri(uri) is True

    @pytest.mark.parametrize(
        "uri",
        [
            "image.png",  # bare relative path
            "/absolute/path.mp4",  # absolute POSIX path
            "file:///etc/passwd",  # file:// stays direct (path-based open)
            "data:image/png;base64,iVBOR...",  # data URI (special case)
            "C:/Windows/path.mp4",  # Windows-style path (no scheme)
        ],
    )
    def test_non_cloud_schemes(self, uri):
        assert is_cloud_uri(uri) is False

    def test_unknown_scheme_is_cloud_open_set(self):
        # Open-set: any scheme not in _DIRECT_URI_SCHEMES (and not a bare
        # path) routes to fsspec — even unfamiliar / future ones. Whether
        # the call succeeds at runtime is fsspec's responsibility.
        assert is_cloud_uri("custom-scheme://thing") is True
        assert is_cloud_uri("webdav://host/path") is True
        assert is_cloud_uri("gdrive://folder/file.png") is True

    def test_direct_schemes_excluded_from_cloud(self):
        # MediaRef handles these schemes directly (without fsspec).
        assert _DIRECT_URI_SCHEMES == frozenset({"file", "data"})


class TestMediaRefCloudUri:
    def test_property_true_for_s3(self):
        assert MediaRef(uri="s3://bucket/clip.mp4", pts_ns=0).is_cloud_uri is True

    def test_property_false_for_local(self):
        assert MediaRef(uri="image.png").is_cloud_uri is False

    def test_property_true_for_http(self):
        # http(s) goes through fsspec's HTTPFileSystem under the unified dispatch.
        assert MediaRef(uri="https://example.com/x.png").is_cloud_uri is True
        assert MediaRef(uri="http://example.com/x.png").is_cloud_uri is True

    def test_is_relative_path_false_for_cloud(self):
        # Cloud URIs are absolute by construction.
        assert MediaRef(uri="s3://bucket/k.png").is_relative_path is False

    def test_validate_uri_uses_fsspec(self):
        _put_bytes("memory://validation/present.png", b"data")

        assert MediaRef(uri="memory://validation/present.png").validate_uri()
        assert not MediaRef(uri="memory://validation/missing.png").validate_uri()

    def test_resolve_relative_path_warns_for_cloud(self):
        ref = MediaRef(uri="s3://bucket/k.png")
        with pytest.warns(UserWarning, match="cloud"):
            out = ref.resolve_relative_path("/data")
        assert out is ref

    def test_resolve_relative_path_error_for_cloud(self):
        ref = MediaRef(uri="s3://bucket/k.png")
        with pytest.raises(ValueError, match="cloud"):
            ref.resolve_relative_path("/data", on_unresolvable="error")

    def test_resolve_relative_path_ignore_for_cloud(self):
        ref = MediaRef(uri="s3://bucket/k.png")
        # "ignore" must not warn or raise.
        out = ref.resolve_relative_path("/data", on_unresolvable="ignore")
        assert out is ref


# ---------------------------------------------------------------------------
# Image loading via memory://
# ---------------------------------------------------------------------------


class TestImageLoadingFromMemoryFS:
    def test_load_png_from_memory_uri(self, sample_rgb_array):
        _put_bytes("memory://imgs/red.png", _png_bytes(sample_rgb_array))
        ref = MediaRef(uri="memory://imgs/red.png")
        loaded = ref.to_ndarray(format="rgb")
        assert loaded.shape == sample_rgb_array.shape
        # PNG is lossless: compare exactly.
        np.testing.assert_array_equal(loaded, sample_rgb_array)

    def test_to_pil_image_from_memory_uri(self, sample_rgb_array):
        _put_bytes("memory://imgs/red.png", _png_bytes(sample_rgb_array))
        ref = MediaRef(uri="memory://imgs/red.png")
        pil = ref.to_pil_image()
        assert pil.size == (sample_rgb_array.shape[1], sample_rgb_array.shape[0])

    def test_loading_from_missing_memory_uri_errors(self):
        ref = MediaRef(uri="memory://does/not/exist.png")
        with pytest.raises((ValueError, FileNotFoundError)):
            ref.to_ndarray()

    def test_storage_options_are_forwarded(self, sample_rgb_array, monkeypatch):
        _put_bytes("memory://imgs/options.png", _png_bytes(sample_rgb_array))
        import mediaref._internal as internal_mod

        real_open = internal_mod.fsspec.open
        calls = []

        def spy(*args, **kwargs):
            calls.append(kwargs.copy())
            return real_open(*args, **kwargs)

        monkeypatch.setattr(internal_mod.fsspec, "open", spy)
        loaded = MediaRef(uri="memory://imgs/options.png").to_ndarray(storage_options={"test_option": "image"})

        assert loaded.shape == sample_rgb_array.shape
        assert calls[-1]["test_option"] == "image"


# ---------------------------------------------------------------------------
# Video loading via memory:// (TensorCodec)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_video_bytes(sample_video_file) -> tuple[bytes, list[int]]:
    path, timestamps = sample_video_file
    return path.read_bytes(), timestamps


@pytest.mark.tensorcodec
class TestVideoLoadingFromMemoryFS:
    def test_to_ndarray_single_frame_from_memory_uri(self, sample_video_bytes):
        data, pts_ns_list = sample_video_bytes
        _put_bytes("memory://videos/clip.mp4", data)

        ref = MediaRef(uri="memory://videos/clip.mp4", pts_ns=pts_ns_list[2])
        frame = ref.to_ndarray(format="rgb")
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8

    def test_batch_decode_from_memory_uri(self, sample_video_bytes):
        data, pts_ns_list = sample_video_bytes
        _put_bytes("memory://videos/clip.mp4", data)

        refs = [MediaRef(uri="memory://videos/clip.mp4", pts_ns=t) for t in pts_ns_list]
        frames = batch_decode(refs)
        assert len(frames) == len(refs)
        for f in frames:
            assert f.shape == (48, 64, 3)
            assert f.dtype == np.uint8

    def test_batch_decode_mixed_local_and_cloud(self, sample_video_bytes, tmp_path):
        # Same video served via both memory:// and a local path; batch_decode
        # should handle both groups in one call.
        data, pts_ns_list = sample_video_bytes
        local_path = tmp_path / "clip.mp4"
        local_path.write_bytes(data)
        _put_bytes("memory://videos/clip.mp4", data)

        refs = [
            MediaRef(uri=str(local_path), pts_ns=pts_ns_list[1]),
            MediaRef(uri="memory://videos/clip.mp4", pts_ns=pts_ns_list[3]),
        ]
        frames = batch_decode(refs, allow_multi_video=True)
        assert len(frames) == 2
        assert all(f.shape == (48, 64, 3) for f in frames)

    def test_repeated_decode_same_cloud_uri_reuses_cache(self, sample_video_bytes, monkeypatch):
        """Two batch_decode calls on the same cloud URI must HIT the cache and
        open fsspec exactly once. Proves both the perf claim and the absence
        of the stale-handle bug — a stale handle would raise on the second
        call instead of HITting.
        """
        data, pts_ns_list = sample_video_bytes
        _put_bytes("memory://videos/clip.mp4", data)

        # Spy on fsspec.open as imported by mediaref._internal (the source of
        # open_cloud, which cached_av delegates to for cloud URIs).
        import mediaref._internal as internal_mod

        open_calls = []
        real_open = internal_mod.fsspec.open

        def spy(*args, **kwargs):
            open_calls.append(args[0] if args else kwargs.get("urlpath"))
            return real_open(*args, **kwargs)

        monkeypatch.setattr(internal_mod.fsspec, "open", spy)

        refs = [MediaRef(uri="memory://videos/clip.mp4", pts_ns=pts_ns_list[1])]
        batch_decode(refs)  # 1st call: cache miss → opens fsspec once
        batch_decode(refs)  # 2nd call: cache hit → must NOT reopen

        assert open_calls == ["memory://videos/clip.mp4"], (
            f"expected single fsspec.open(memory://...), got {open_calls}"
        )

    def test_cleanup_cache_closes_owned_filelike(self, sample_video_bytes):
        """``cleanup_cache()`` must dispose owned fsspec file-likes; a fresh
        decode after cleanup must work (no left-over closed-handle state).
        """
        data, pts_ns_list = sample_video_bytes
        _put_bytes("memory://videos/clip.mp4", data)
        refs = [MediaRef(uri="memory://videos/clip.mp4", pts_ns=pts_ns_list[1])]
        batch_decode(refs)
        cleanup_cache()
        # Reopen-and-decode must succeed against a fresh cache entry.
        frames = batch_decode(refs)
        assert len(frames) == 1
        assert frames[0].shape == (48, 64, 3)

    def test_video_storage_options_are_forwarded(self, sample_video_bytes, monkeypatch):
        data, pts_ns_list = sample_video_bytes
        _put_bytes("memory://videos/options.mp4", data)
        import mediaref._internal as internal_mod

        real_open = internal_mod.fsspec.open
        calls = []

        def spy(*args, **kwargs):
            calls.append(kwargs.copy())
            return real_open(*args, **kwargs)

        monkeypatch.setattr(internal_mod.fsspec, "open", spy)
        refs = [MediaRef(uri="memory://videos/options.mp4", pts_ns=pts_ns_list[1])]
        frames = batch_decode(refs, storage_options={"test_option": "video"})

        assert frames[0].shape == (48, 64, 3)
        assert calls[-1]["test_option"] == "video"

    def test_storage_options_are_part_of_video_cache_identity(self, sample_video_bytes, monkeypatch):
        data, pts_ns_list = sample_video_bytes
        _put_bytes("memory://videos/cache-options.mp4", data)
        import mediaref._internal as internal_mod

        real_open = internal_mod.fsspec.open
        calls = []

        def spy(*args, **kwargs):
            calls.append(kwargs.copy())
            return real_open(*args, **kwargs)

        monkeypatch.setattr(internal_mod.fsspec, "open", spy)
        refs = [MediaRef(uri="memory://videos/cache-options.mp4", pts_ns=pts_ns_list[1])]
        batch_decode(refs, storage_options={"token": "first"})
        batch_decode(refs, storage_options={"token": "second"})

        assert [call["token"] for call in calls] == ["first", "second"]


# ---------------------------------------------------------------------------
# Real hf:// integration (network)
# ---------------------------------------------------------------------------


@pytest.mark.tensorcodec
@pytest.mark.network
class TestHfDatasetIntegration:
    """Live integration tests against the open-world-agents/D2E-480p HF dataset.

    These tests verify that the fsspec hf:// dispatch works against a real
    HuggingFace endpoint — not a mock, not memory://. The dataset is one of the
    canonical downstream consumers of MediaRef (see README), so a real-world
    breakage here is a release blocker.

    HF Hub serves byte-range requests, so each test pulls only the bytes it
    actually decodes (a few MB), not the multi-TB dataset.
    """

    # Stable, smallest-game video in D2E-480p. If this file is ever deleted
    # upstream, swap it for any other ``<game>/<id>.mkv`` under the dataset.
    HF_VIDEO_URI = "hf://datasets/open-world-agents/D2E-480p/Apex_Legends/0805_01.mkv"

    @pytest.fixture(scope="class")
    def _hf_available(self):
        pytest.importorskip("huggingface_hub")
        # Ensure the URI is actually reachable before running the decode tests
        # (gives a clearer skip reason than a mid-decode failure).
        import fsspec

        try:
            with fsspec.open(self.HF_VIDEO_URI, "rb") as f:
                f.read(1)
        except Exception as e:
            pytest.skip(f"D2E-480p HF endpoint unreachable: {e}")

    def test_single_frame_decode_from_hf(self, _hf_available):
        ref = MediaRef(uri=self.HF_VIDEO_URI, pts_ns=1_000_000_000)
        rgb = ref.to_ndarray()
        assert rgb.dtype == np.uint8
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        # 480p — height should be ~480 pixels (allow generic check on H>=240)
        assert rgb.shape[0] >= 240, f"unexpected height: {rgb.shape}"

    def test_batch_decode_from_hf(self, _hf_available):
        refs = [MediaRef(uri=self.HF_VIDEO_URI, pts_ns=int(t * 1e9)) for t in (0.5, 1.0, 2.0)]
        frames = batch_decode(refs)
        assert len(frames) == 3
        shapes = {f.shape for f in frames}
        assert len(shapes) == 1, f"frame shapes diverged: {shapes}"
