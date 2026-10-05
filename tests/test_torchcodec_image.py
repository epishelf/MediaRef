"""Tests for optional TorchCodec image decoding and wide-dtype outputs."""

import io
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import PIL.Image
import pytest

from mediaref import MediaRef, batch_decode
from mediaref._internal import load_video_frame_as_rgba
from mediaref.data_uri import DataURI
from mediaref.video_decoder import FrameBatch


def test_missing_torchcodec_image_reports_unified_extra(sample_image_file, monkeypatch):
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)
    with pytest.raises(ImportError, match=r"mediaref\[torchcodec\]"):
        MediaRef(uri=str(sample_image_file)).to_ndarray(image_decoder="torchcodec")


@pytest.fixture
def fake_torchcodec_image(monkeypatch: pytest.MonkeyPatch):
    calls = []
    torchcodec = ModuleType("torchcodec")
    decoders = ModuleType("torchcodec.decoders")

    def decode_image(source, **options):
        calls.append((source, options))
        if isinstance(source, (str, Path)):
            image = PIL.Image.open(source)
        else:
            image = PIL.Image.open(io.BytesIO(source))
        rgba = np.asarray(image.convert("RGBA"))
        if options.get("output_dtype") == "auto":
            rgba = rgba.astype(np.uint16) * 257
        return np.moveaxis(rgba, -1, 0)

    decoders.decode_image = decode_image  # type: ignore[attr-defined]
    torchcodec.decoders = decoders  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchcodec", torchcodec)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    return calls


def test_torchcodec_image_local_path(sample_image_file: Path, fake_torchcodec_image):
    rgb = MediaRef(uri=str(sample_image_file)).to_ndarray(image_decoder="torchcodec")

    assert rgb.shape == (48, 64, 3)
    assert rgb.dtype == np.uint8
    assert fake_torchcodec_image[0][1] == {"mode": "RGB_ALPHA"}


def test_torchcodec_image_data_uri(sample_image_file: Path, fake_torchcodec_image):
    data_uri = DataURI.from_file(sample_image_file)

    MediaRef(uri=data_uri.uri).to_ndarray(image_decoder="torchcodec")

    assert isinstance(fake_torchcodec_image[0][0], bytes)


def test_torchcodec_image_reads_fsspec_source(sample_image_file: Path, fake_torchcodec_image):
    encoded = sample_image_file.read_bytes()
    opened = type("Opened", (), {"__enter__": lambda self: io.BytesIO(encoded), "__exit__": lambda *args: None})()

    with patch("mediaref._internal.open_cloud", return_value=opened) as open_cloud:
        MediaRef(uri="memory://bucket/image.png").to_ndarray(
            image_decoder="torchcodec",
            storage_options={"token": "secret"},
        )

    open_cloud.assert_called_once_with("memory://bucket/image.png", storage_options={"token": "secret"})
    assert fake_torchcodec_image[0][0] == encoded


def test_torchcodec_image_preserves_uint16_output(sample_image_file: Path, fake_torchcodec_image):
    rgba = MediaRef(uri=str(sample_image_file)).to_ndarray(
        format="rgba",
        image_decoder="torchcodec",
        image_decoder_options={"output_dtype": "auto"},
    )

    assert rgba.dtype == np.uint16
    assert np.all(rgba[..., 3] == np.iinfo(np.uint16).max)


def test_torchcodec_image_applies_exif_orientation(tmp_path: Path, fake_torchcodec_image):
    source = tmp_path / "oriented.jpg"
    image = PIL.Image.new("RGB", (4, 2), (20, 40, 60))
    exif = PIL.Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)

    default = MediaRef(uri=str(source)).to_ndarray()
    torchcodec = MediaRef(uri=str(source)).to_ndarray(image_decoder="torchcodec")

    np.testing.assert_array_equal(torchcodec, default)
    assert torchcodec.shape == (4, 2, 3)


def test_torchcodec_image_mode_is_owned_by_mediaref(sample_image_file: Path, fake_torchcodec_image):
    with pytest.raises(ValueError, match="controls the image decoder's mode"):
        MediaRef(uri=str(sample_image_file)).to_ndarray(
            image_decoder="torchcodec",
            image_decoder_options={"mode": "GRAY"},
        )


def test_torchcodec_image_batch_forwards_options(sample_image_file: Path, fake_torchcodec_image):
    refs = [MediaRef(uri=str(sample_image_file))]

    images = batch_decode(
        refs,
        allow_images=True,
        image_decoder="torchcodec",
        image_decoder_options={"output_dtype": "auto"},
    )

    assert images[0].dtype == np.uint16
    assert fake_torchcodec_image[0][1]["output_dtype"] == "auto"


def test_to_pil_image_rejects_wide_dtype(sample_image_file: Path, fake_torchcodec_image):
    with pytest.raises(ValueError, match="requires uint8 output"):
        MediaRef(uri=str(sample_image_file)).to_pil_image(
            image_decoder="torchcodec",
            image_decoder_options={"output_dtype": "auto"},
        )


def test_float_video_output_gets_unit_alpha(sample_video_file: tuple[Path, list[int]]):
    import mediaref.video_decoder as video_decoder

    video_path, timestamps = sample_video_file

    class FloatDecoder:
        def __init__(self, source, **options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_frames_played_at(self, seconds):
            return FrameBatch(
                data=np.full((1, 3, 2, 3), 0.5, dtype=np.float32),
                pts_seconds=np.asarray(seconds),
                duration_seconds=np.asarray([0.1]),
            )

    with patch.dict(video_decoder.__dict__, {"TorchCodecVideoDecoder": FloatDecoder}):
        rgba = load_video_frame_as_rgba(str(video_path), timestamps[0], decoder="torchcodec")

    assert rgba.dtype == np.float32
    assert np.all(rgba[..., :3] == 0.5)
    assert np.all(rgba[..., 3] == 1.0)
