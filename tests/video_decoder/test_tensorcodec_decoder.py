"""Default video integration tests independent of PyAV and Torch."""

import io
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
import pytest

pytest.importorskip("tensorcodec_av")


@pytest.fixture(scope="module")
def codec_videos(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg CLI needed to generate independent fixtures")
    root = tmp_path_factory.mktemp("tensorcodec-video")
    paths = {}
    for name in ("cfr", "vfr", "offset"):
        encoded = root / f"{name}-encoded.mp4"
        args = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10"]
        if name == "vfr":
            args += ["-vf", "setpts=if(lt(N\\,5)\\,N\\,2*N-5)/(10*TB)"]
        if name == "offset":
            args += ["-output_ts_offset", "2"]
        args += [
            "-frames:v",
            "21",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-g",
            "10",
            "-bf",
            "0" if name == "vfr" else "2",
            str(encoded),
        ]
        subprocess.run(args, check=True)
        path = root / f"{name}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-copyts",
                "-i",
                str(encoded),
                "-c:v",
                "copy",
                "-frames:v",
                "20",
                str(path),
            ],
            check=True,
        )
        paths[name] = path
    return paths


@pytest.fixture(autouse=True)
def clear_decoders():
    from mediaref import cleanup_cache

    cleanup_cache()
    yield
    cleanup_cache()


@pytest.mark.parametrize("dimension_order", ["NCHW", "NHWC"])
def test_playback_order_duplicates_and_empty(codec_videos, dimension_order):
    from mediaref.video_decoder import TensorCodecVideoDecoder

    with TensorCodecVideoDecoder(codec_videos["cfr"], dimension_order=dimension_order) as decoder:
        batch = decoder.get_frames_played_at([0.35, 0.01, 0.35, 1.99])
        np.testing.assert_allclose(batch.pts_seconds, [0.3, 0, 0.3, 1.9])
        assert batch.data.shape == (4, 3, 48, 64)
        np.testing.assert_array_equal(batch.data[0], batch.data[2])
        assert decoder.get_frames_played_at([]).data.shape == (0, 3, 48, 64)
        saved = batch.data.copy()
    TensorCodecVideoDecoder.clear_cache()
    np.testing.assert_array_equal(batch.data, saved)


def test_vfr_offset_ranges_and_indices(codec_videos):
    from mediaref.video_decoder import TensorCodecVideoDecoder

    with TensorCodecVideoDecoder(codec_videos["vfr"]) as decoder:
        batch = decoder.get_frames_played_at([0.65, 0.45, 0.65])
        np.testing.assert_allclose(batch.pts_seconds, [0.5, 0.4, 0.5])
        np.testing.assert_allclose(batch.duration_seconds, [0.2, 0.1, 0.2])
        selected = decoder.get_frames_played_in_range(0.45, 1.0)
        np.testing.assert_allclose(selected.pts_seconds, [0.4, 0.5, 0.7, 0.9])
        grid = decoder.get_frames_played_in_range(0.5, 1.0, fps=4)
        np.testing.assert_allclose(grid.pts_seconds, [0.5, 0.75])
        np.testing.assert_array_equal(decoder.get_frames_at([5, 4, 5]).data, batch.data)
    with TensorCodecVideoDecoder(codec_videos["offset"]) as decoder:
        assert decoder.metadata.begin_stream_seconds == 2
        np.testing.assert_allclose(decoder.get_frames_played_at([2.35, 2.05]).pts_seconds, [2.3, 2])
        with pytest.raises(RuntimeError, match="outside"):
            decoder.get_frames_played_at([0])


def test_default_public_api_and_native_formats(codec_videos):
    from mediaref import MediaRef, batch_decode

    refs = [MediaRef(uri=str(codec_videos["cfr"]), pts_ns=t) for t in (350_000_000, 10_000_000, 350_000_000)]
    frames = batch_decode(refs, gap_threshold=0.01)
    for ref, frame in zip(refs, frames):
        np.testing.assert_array_equal(frame, ref.to_ndarray())
        assert frame.shape == (48, 64, 3)
    explicit_rgb = batch_decode(refs, decoder_options={"output_format": "rgb"})
    np.testing.assert_array_equal(explicit_rgb[0], frames[0])
    np.testing.assert_array_equal(refs[0].to_ndarray(decoder_options={"output_format": "rgb"}), frames[0])
    assert refs[0].to_pil_image().size == (64, 48)
    with pytest.raises(ValueError, match="does not support pixel format 'yuv420p'"):
        refs[0].to_ndarray(format="native")  # TensorCodec preserves gray and packed RGB(A) samples, not YUV


def test_cache_leases_fsspec_and_owned_sources(codec_videos):
    from mediaref.video_decoder import TensorCodecVideoDecoder

    uri = "memory://tensorcodec/video.mp4"
    with fsspec.open(uri, "wb") as file:
        file.write(codec_videos["cfr"].read_bytes())
    first = TensorCodecVideoDecoder(uri)
    second = TensorCodecVideoDecoder(uri)
    assert first._state is second._state
    state = first._state
    first.close()
    first.close()
    assert second.get_frames_played_at([0]).data.shape == (1, 3, 48, 64)
    second.close()
    assert not state.disposed
    TensorCodecVideoDecoder.clear_cache()
    assert state.disposed
    assert state.owned_open_context is None
    with pytest.raises(RuntimeError, match="closed"):
        first.get_frames_played_at([0])
    source = io.BytesIO(codec_videos["cfr"].read_bytes())
    with TensorCodecVideoDecoder(source) as decoder:
        decoder.get_frames_played_at([0])
        state = decoder._state
    assert state.disposed
    assert not source.closed


def test_clear_active_lease_and_shared_concurrency(codec_videos):
    from mediaref.video_decoder import TensorCodecVideoDecoder

    first = TensorCodecVideoDecoder(codec_videos["cfr"])
    second = TensorCodecVideoDecoder(codec_videos["cfr"])
    TensorCodecVideoDecoder.clear_cache()
    with ThreadPoolExecutor(max_workers=2) as pool:
        batches = list(pool.map(lambda d: d.get_frames_played_at([0.35, 0.01]), [first, second]))
    np.testing.assert_array_equal(batches[0].data, batches[1].data)
    first.close()
    assert second.get_frames_played_at([0]).data.shape[0] == 1
    second.close()


def test_default_decode_needs_neither_pyav_nor_torch(codec_videos):
    script = r"""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'av', 'torch', 'torchcodec'}:
            raise ImportError('forbidden: ' + fullname)
sys.meta_path.insert(0, Block())
from mediaref import MediaRef, batch_decode, cleanup_cache
refs = [MediaRef(uri=sys.argv[1], pts_ns=i) for i in [350000000, 10000000, 350000000]]
assert len(batch_decode(refs)) == 3
assert refs[0].to_ndarray().shape == (48, 64, 3)
cleanup_cache()
assert not {'av', 'torch', 'torchcodec'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, str(codec_videos["cfr"])], check=True, timeout=30)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork requires POSIX")
@pytest.mark.parametrize("backend", ["TensorCodecVideoDecoder", "TorchCodecVideoDecoder"])
def test_fork_reopens_inherited_lease_without_closing_parent(codec_videos, backend):
    from tests import TORCHCODEC_AVAILABLE

    if backend == "TorchCodecVideoDecoder" and not TORCHCODEC_AVAILABLE:
        pytest.skip("Optional TorchCodec video runtime is unavailable")
    script = r"""
import io, multiprocessing, sys
from pathlib import Path
import mediaref.video_decoder as decoders
from mediaref import MediaRef, batch_decode
Decoder = getattr(decoders, sys.argv[2])
path = sys.argv[1]
lease = Decoder(path, num_ffmpeg_threads=2)
lease.get_frames_played_at([0.35])
uncached = Decoder(io.BytesIO(Path(path).read_bytes()), num_ffmpeg_threads=2)
uncached.get_frames_played_at([0.35])
def child(pipe):
    assert uncached.get_frames_played_at([0.35]).pts_seconds.tolist() == [0.3]
    uncached.close()
    pipe.send(lease.get_frames_played_at([0.35]).pts_seconds.tolist())
    assert batch_decode([MediaRef(uri=path, pts_ns=10000000)])[0].shape == (48, 64, 3)
    lease.close()
    pipe.close()
ctx = multiprocessing.get_context('fork')
parent, remote = ctx.Pipe(duplex=False)
process = ctx.Process(target=child, args=(remote,))
process.start()
remote.close()
try:
    assert parent.poll(10), 'child decode hung'
    assert parent.recv() == [0.3]
    process.join(10)
    assert process.exitcode == 0
finally:
    if process.is_alive():
        process.kill()
        process.join()
assert lease.get_frames_played_at([0.35]).pts_seconds.tolist() == [0.3]
assert uncached.get_frames_played_at([0.35]).pts_seconds.tolist() == [0.3]
uncached.close()
lease.close()
"""
    subprocess.run([sys.executable, "-c", script, str(codec_videos["cfr"]), backend], check=True, timeout=30)


def test_options_isolate_cache_and_preserve_float_rgb(codec_videos):
    from mediaref import MediaRef, batch_decode
    from mediaref.video_decoder import TensorCodecVideoDecoder

    path = codec_videos["cfr"]
    with TensorCodecVideoDecoder(path) as integer, TensorCodecVideoDecoder(path, output_dtype=np.float32) as floating:
        assert integer._state is not floating._state
        assert floating.get_frames_played_at([0]).data.dtype == np.float32
    ref = MediaRef(uri=str(path), pts_ns=0)
    rgba = ref.to_ndarray(format="rgba", decoder_options={"output_dtype": np.float32})
    assert rgba.dtype == np.float32
    np.testing.assert_array_equal(rgba[..., 3], 1)
    with pytest.raises(ValueError, match="CPU"):
        batch_decode([ref], decoder_options={"device": "cuda"})


@pytest.mark.parametrize("name", ["cfr", "vfr", "offset"])
def test_agreement_with_optional_torchcodec(codec_videos, name):
    from tests import TORCHCODEC_AVAILABLE

    if not TORCHCODEC_AVAILABLE:
        pytest.skip("Optional TorchCodec video runtime is unavailable")
    from mediaref.video_decoder import TensorCodecVideoDecoder, TorchCodecVideoDecoder

    start = 2 if name == "offset" else 0
    queries = [start + 0.35, start + 0.01, start + 0.35, start + 1.05]
    with (
        TensorCodecVideoDecoder(codec_videos[name]) as decoder,
        TorchCodecVideoDecoder(codec_videos[name], num_ffmpeg_threads=1) as oracle,
    ):
        actual = decoder.get_frames_played_at(queries)
        expected = oracle.get_frames_played_at(queries)
        np.testing.assert_allclose(actual.pts_seconds, expected.pts_seconds, atol=1e-12)
        np.testing.assert_allclose(actual.duration_seconds, expected.duration_seconds, atol=1e-12)
        difference = np.abs(actual.data.astype(np.int16) - expected.data.astype(np.int16))
        assert difference.max() <= 1


def test_cleanup_error_cannot_loop_on_disposed_cache_state(codec_videos, monkeypatch):
    from mediaref.video_decoder import TensorCodecVideoDecoder

    class FailingContext:
        def __exit__(self, *args):
            raise OSError("source cleanup failed")

    lease = TensorCodecVideoDecoder(codec_videos["cfr"])
    state = lease._state
    state.owned_open_context = FailingContext()
    with pytest.raises(OSError, match="source cleanup failed"):
        TensorCodecVideoDecoder.clear_cache()
    assert state.disposed and state.owned_open_context is None
    acquire = lease._acquire_state
    calls = 0

    def bounded_acquire():
        nonlocal calls
        calls += 1
        assert calls <= 1, "disposed cache state retried indefinitely"
        return acquire()

    monkeypatch.setattr(lease, "_acquire_state", bounded_acquire)
    with pytest.raises(RuntimeError, match="retry cleanup_cache"):
        lease.get_frames_played_at([0])
    TensorCodecVideoDecoder.clear_cache()
    monkeypatch.setattr(lease, "_acquire_state", acquire)
    assert lease.get_frames_played_at([0]).data.shape == (1, 3, 48, 64)
    lease.close()
