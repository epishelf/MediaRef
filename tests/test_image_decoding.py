"""Tests for the default TensorCodec image decoder."""

from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageOps
import pytest

from mediaref import MediaRef, batch_decode
from mediaref.data_uri import DataURI


def _pattern(height: int = 6, width: int = 10) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return np.stack([x * 255 // width, y * 255 // height, (x + y) * 127 // (height + width)], axis=-1).astype(np.uint8)


def _save(image: PIL.Image.Image, path: Path, **kwargs) -> Path:
    image.save(path, **kwargs)
    return path


def test_png_rgba_is_lossless(tmp_path: Path):
    rgba = np.dstack([_pattern(), np.arange(60, dtype=np.uint8).reshape(6, 10) * 4])
    path = _save(PIL.Image.fromarray(rgba, "RGBA"), tmp_path / "alpha.png")

    np.testing.assert_array_equal(MediaRef(uri=str(path)).to_ndarray(format="rgba"), rgba)


def test_rgb_png_gets_opaque_alpha(tmp_path: Path):
    rgb = _pattern()
    path = _save(PIL.Image.fromarray(rgb, "RGB"), tmp_path / "rgb.png")

    rgba = MediaRef(uri=str(path)).to_ndarray(format="rgba")
    assert rgba.dtype == np.uint8
    np.testing.assert_array_equal(rgba[..., :3], rgb)
    assert np.all(rgba[..., 3] == 255)


def test_grayscale_png_expands_to_rgb(tmp_path: Path):
    gray = _pattern()[..., 0]
    path = _save(PIL.Image.fromarray(gray, "L"), tmp_path / "gray.png")

    rgba = MediaRef(uri=str(path)).to_ndarray(format="rgba")
    np.testing.assert_array_equal(rgba[..., :3], np.repeat(gray[..., None], 3, axis=-1))
    assert np.all(rgba[..., 3] == 255)


def test_palette_png_matches_pillow(tmp_path: Path):
    image = PIL.Image.fromarray(_pattern(), "RGB").convert("P")
    path = _save(image, tmp_path / "palette.png")

    expected = np.asarray(image.convert("RGBA"))
    np.testing.assert_array_equal(MediaRef(uri=str(path)).to_ndarray(format="rgba"), expected)


def test_jpeg_matches_pillow(tmp_path: Path):
    path = _save(PIL.Image.fromarray(_pattern(32, 48), "RGB"), tmp_path / "image.jpg", quality=95)

    with PIL.Image.open(path) as source:
        expected = np.asarray(source.convert("RGB"))
    decoded = MediaRef(uri=str(path)).to_ndarray()
    assert decoded.shape == expected.shape and decoded.dtype == np.uint8
    assert np.abs(decoded.astype(int) - expected.astype(int)).max() <= 2


@pytest.mark.parametrize("orientation", range(1, 9))
def test_jpeg_exif_orientation_matches_pillow(tmp_path: Path, orientation: int):
    image = PIL.Image.fromarray(_pattern(16, 24), "RGB")
    exif = PIL.Image.Exif()
    exif[274] = orientation
    path = _save(image, tmp_path / "oriented.jpg", exif=exif, quality=100)

    with PIL.Image.open(path) as source:
        expected = np.asarray(PIL.ImageOps.exif_transpose(source).convert("RGB"))
    decoded = MediaRef(uri=str(path)).to_ndarray()
    assert decoded.shape == expected.shape
    assert np.abs(decoded.astype(int) - expected.astype(int)).max() <= 2


def test_lossless_webp_with_alpha(tmp_path: Path):
    rgba = np.dstack([_pattern(), np.full((6, 10), 128, np.uint8)])
    path = _save(PIL.Image.fromarray(rgba, "RGBA"), tmp_path / "image.webp", lossless=True)

    np.testing.assert_array_equal(MediaRef(uri=str(path)).to_ndarray(format="rgba"), rgba)


def test_animated_gif_returns_first_frame(tmp_path: Path):
    frames = [PIL.Image.new("RGB", (8, 4), color) for color in ((255, 0, 0), (0, 0, 255))]
    path = tmp_path / "animated.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)

    rgb = MediaRef(uri=str(path)).to_ndarray()
    assert rgb.shape == (4, 8, 3)
    np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])


def test_16bit_png_scales_to_uint8_or_keeps_uint16(tmp_path: Path):
    gray16 = (np.arange(60, dtype=np.uint16).reshape(6, 10) * 1000).astype(np.uint16)
    path = tmp_path / "gray16.png"
    PIL.Image.frombytes("I;16", (gray16.shape[1], gray16.shape[0]), gray16.astype("<u2").tobytes()).save(path)
    ref = MediaRef(uri=str(path))

    scaled = ref.to_ndarray(format="gray")
    assert scaled.dtype == np.uint8
    np.testing.assert_array_equal(scaled, np.rint(gray16 / 257).astype(np.uint8))

    wide = ref.to_ndarray(format="rgba", image_decoder_options={"output_dtype": "auto"})
    assert wide.dtype == np.uint16
    np.testing.assert_array_equal(wide[..., 0], gray16)


def test_data_uri_decodes_through_tensorcodec():
    rgb = _pattern()
    data_uri = DataURI.from_image(rgb, format="png")

    np.testing.assert_array_equal(data_uri.to_ndarray(), rgb)
    np.testing.assert_array_equal(MediaRef(uri=data_uri.uri).to_ndarray(), rgb)


def test_batch_decode_images(tmp_path: Path):
    rgb = _pattern()
    path = _save(PIL.Image.fromarray(rgb, "RGB"), tmp_path / "rgb.png")

    (decoded,) = batch_decode([MediaRef(uri=str(path))], allow_images=True)
    np.testing.assert_array_equal(decoded, rgb)


def test_unknown_image_decoder_is_rejected(sample_image_file: Path):
    with pytest.raises(ValueError, match="Must be 'tensorcodec' or 'torchcodec'"):
        MediaRef(uri=str(sample_image_file)).to_ndarray(image_decoder="pillow")  # type: ignore[arg-type]


def test_image_mode_is_owned_by_mediaref(sample_image_file: Path):
    with pytest.raises(ValueError, match="controls the image decoder's mode"):
        MediaRef(uri=str(sample_image_file)).to_ndarray(image_decoder_options={"mode": "GRAY"})


def test_undecodable_bytes_raise_value_error(tmp_path: Path):
    path = tmp_path / "junk.png"
    path.write_bytes(b"not an image")

    with pytest.raises(ValueError, match="Failed to load image"):
        MediaRef(uri=str(path)).to_ndarray()
