from fractions import Fraction

import numpy as np
import pytest

av = pytest.importorskip("av")

from mediaref import MediaRef, batch_decode  # noqa: E402
from mediaref.video_decoder import PyAVVideoDecoder  # noqa: E402


@pytest.fixture(params=["pyav", "tensorcodec"])
def backend(request):
    if request.param == "tensorcodec":
        pytest.importorskip("tensorcodec_av")
    return request.param


def _decoder_class(backend):
    if backend == "pyav":
        return PyAVVideoDecoder
    from mediaref.video_decoder import TensorCodecVideoDecoder

    return TensorCodecVideoDecoder


@pytest.fixture(params=["gray12le", "gray16le", "gray16be"])
def native_video(tmp_path, request):
    pixel_format = request.param
    path = tmp_path / "native.nut"
    expected = np.arange(256, dtype=np.uint16).reshape(16, 16) * (7 if pixel_format == "gray12le" else 127)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("rawvideo", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = pixel_format
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(expected + i, format=pixel_format)
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path, pixel_format, expected


def test_native_values_playback(native_video, backend):
    path, fmt, expected = native_video
    decoder_class = _decoder_class(backend)
    with decoder_class(path, output_format="native", expected_pixel_format=fmt) as decoder:
        batch = decoder.get_frames_played_at([0.19, 0.0, 0.19])
        assert batch.pixel_format == fmt
        assert batch.data.dtype == np.uint16
        assert batch.data.shape == (3, 1, 16, 16)
        np.testing.assert_array_equal(batch.data[:, 0], [expected + 1, expected, expected + 1])
        empty = decoder.get_frames_played_at([])
        assert empty.data.shape == (0, 1, 16, 16) and empty.data.dtype == np.uint16
        selected = decoder.get_frames_played_in_range(0, 0.15)
        np.testing.assert_array_equal(selected.data[:, 0], [expected, expected + 1])
    with pytest.raises((ValueError, RuntimeError), match="(?i)expected"):
        with decoder_class(path, output_format="native", expected_pixel_format="gray") as decoder:
            decoder.get_frames_played_at([0])


def test_native_unsupported_format(tmp_path):
    path = tmp_path / "rgb.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
        for packet in [*stream.encode(frame), *stream.encode()]:
            container.mux(packet)
    with PyAVVideoDecoder(path, output_format="native") as decoder:
        with pytest.raises(ValueError, match="does not support"):
            decoder.get_frames_played_at([0])
    with PyAVVideoDecoder(path) as decoder:
        assert decoder.get_frames_played_at([0]).data.shape == (1, 3, 16, 16)


def test_public_native_api(native_video, backend):
    path, fmt, expected = native_video
    refs = [MediaRef(uri=str(path), pts_ns=t) for t in [190_000_000, 0, 190_000_000]]
    options = {"expected_pixel_format": fmt}
    frames = batch_decode(refs, output_format="native", decoder=backend, decoder_options=options)
    for ref, frame, value in zip(refs, frames, [expected + 1, expected, expected + 1]):
        assert frame.dtype == np.uint16 and frame.shape == (16, 16)
        np.testing.assert_array_equal(frame, value)
        np.testing.assert_array_equal(ref.to_ndarray(format="native", decoder=backend, decoder_options=options), value)
    frames[0][:] = 0
    np.testing.assert_array_equal(frames[2], expected + 1)
    assert refs[0].to_ndarray(decoder=backend).shape == (16, 16, 3)
    with pytest.raises(ValueError, match="PyAV or TensorCodec"):
        refs[0].to_ndarray(format="native", decoder="torchcodec")
    with pytest.raises((ValueError, RuntimeError), match="(?i)expected"):
        refs[0].to_ndarray(format="native", decoder=backend, decoder_options={"expected_pixel_format": "gray"})
    with pytest.raises(ValueError, match="Conflicting"):
        batch_decode(refs, output_format="native", decoder=backend, decoder_options={"output_format": "rgb"})
    with pytest.raises(ValueError, match="format='native'"):
        refs[0].to_ndarray(decoder_options={"output_format": "native"})


def test_native_images_explicitly_unsupported():
    with pytest.raises(ValueError, match="video refs only"):
        MediaRef(uri="not-opened.png").to_ndarray(format="native")


@pytest.mark.parametrize("fmt,channels", [("gray", 1), ("rgb24", 3), ("rgba", 4)])
def test_native_byte_formats(tmp_path, fmt, channels, backend):
    path = tmp_path / "pixels.nut"
    shape = (16, 16) if channels == 1 else (16, 16, channels)
    expected = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("rawvideo", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = fmt
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(expected, format=fmt)
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    ref = MediaRef(uri=str(path), pts_ns=0)
    actual = ref.to_ndarray(format="native", decoder=backend)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_codec_backends_without_native_output_refuse_it(native_video):
    """TorchCodec decodes to RGB only: native output is refused rather than silently converted."""
    pytest.importorskip("torchcodec")
    from mediaref.video_decoder import TorchCodecVideoDecoder

    path, _, _ = native_video
    with pytest.raises(ValueError, match="PyAV or TensorCodec"):
        TorchCodecVideoDecoder(path, output_format="native")


def test_expected_pixel_format_requires_native_output(native_video, backend):
    path, fmt, _ = native_video
    with pytest.raises(ValueError, match="requires native output"):
        _decoder_class(backend)(path, expected_pixel_format=fmt)
