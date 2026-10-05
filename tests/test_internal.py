"""Tests for internal utility functions.

These tests verify internal implementation details not exposed by the public API:
- Internal RGBA format handling (load_image_as_rgba)
"""

import hashlib

import numpy as np
import numpy.typing as npt
import pytest

from mediaref._internal import load_image_as_rgba, rgba_to_format
from mediaref.data_uri import DataURI


class TestInternalRGBAHandling:
    """Test internal RGBA format handling.

    These tests verify that load_image_as_rgba() correctly handles different
    image formats and maintains data integrity through encode/decode cycles.
    """

    def test_load_png_as_rgba_lossless(self, sample_rgba_array: npt.NDArray[np.uint8]):
        """Test that PNG encoding/decoding via load_image_as_rgba is lossless."""
        # Create data URI from RGBA array
        rgb_array = sample_rgba_array[..., :3]
        data_uri = DataURI.from_image(rgb_array, format="png").uri

        # Decode using internal function
        decoded_rgba = load_image_as_rgba(data_uri)

        # Verify lossless roundtrip
        assert decoded_rgba.shape == sample_rgba_array.shape
        assert decoded_rgba.dtype == sample_rgba_array.dtype
        np.testing.assert_array_equal(decoded_rgba, sample_rgba_array)

    def test_load_bmp_as_rgba_lossless(self, sample_rgba_array: npt.NDArray[np.uint8]):
        """Test that BMP encoding/decoding via load_image_as_rgba is lossless."""
        # Create data URI from RGBA array
        rgb_array = sample_rgba_array[..., :3]
        data_uri = DataURI.from_image(rgb_array, format="bmp").uri

        # Decode using internal function
        decoded_rgba = load_image_as_rgba(data_uri)

        # Verify lossless roundtrip
        assert decoded_rgba.shape == sample_rgba_array.shape
        assert decoded_rgba.dtype == sample_rgba_array.dtype
        np.testing.assert_array_equal(decoded_rgba, sample_rgba_array)

    def test_load_jpeg_as_rgba_lossy(self, sample_rgba_array: npt.NDArray[np.uint8]):
        """Test that JPEG encoding/decoding via load_image_as_rgba handles lossy compression."""
        # Create data URI from RGBA array
        rgb_array = sample_rgba_array[..., :3]
        data_uri = DataURI.from_image(rgb_array, format="jpeg", quality=85).uri

        # Decode using internal function
        decoded_rgba = load_image_as_rgba(data_uri)

        # Verify shape and dtype
        assert decoded_rgba.shape == sample_rgba_array.shape
        assert decoded_rgba.dtype == sample_rgba_array.dtype

        # JPEG is lossy - verify arrays are similar but not identical
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(decoded_rgba, sample_rgba_array)
        assert np.abs(decoded_rgba.astype(float) - sample_rgba_array.astype(float)).mean() < 50


class TestColorConversion:
    """``rgba_to_format`` matches the ``cv2.cvtColor`` output it replaced (expected values from OpenCV 4.14)."""

    PIXELS = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [1, 2, 3]]
    PIXELS += [[128, 64, 32], [17, 200, 99], [250, 5, 130], [3, 3, 4], [77, 77, 78], [201, 99, 12]]

    @staticmethod
    def _rgba(rgb, dtype):
        rgb = np.asarray(rgb, dtype=dtype)
        alpha = np.full((len(rgb), 1), np.iinfo(dtype).max, dtype=dtype)
        return np.concatenate([rgb, alpha], axis=1)[None]

    def test_gray_uint8_matches_opencv(self):
        gray = rgba_to_format(self._rgba(self.PIXELS, np.uint8), "gray")
        assert gray.dtype == np.uint8
        assert gray.tolist() == [[0, 255, 76, 150, 29, 2, 79, 134, 93, 3, 77, 120]]

    def test_gray_uint16_matches_opencv(self):
        pixels = [[0, 0, 0], [6, 6, 6], [13, 14, 14], [21, 20, 21], [28, 28, 27], [292, 549, 806]]
        pixels += [[32938, 16490, 8266], [4418, 51449, 25492], [64306, 1341, 33466], [834, 834, 1091]]
        pixels += [[19859, 19859, 20116], [51734, 25520, 3161]]
        gray = rgba_to_format(self._rgba(pixels, np.uint16), "gray")
        assert gray.dtype == np.uint16
        assert gray.tolist() == [[0, 6, 14, 20, 28, 501, 20471, 34428, 23830, 863, 19888, 30810]]

    def test_gray_random_uint8_matches_opencv(self):
        rgba = np.random.default_rng(1234).integers(0, 256, (64, 64, 4), dtype=np.uint8)
        gray = rgba_to_format(rgba, "gray")
        assert hashlib.sha256(gray.tobytes()).hexdigest() == (
            "d7fc8db69e82e04ec6f5026a65cbc28dd721d050fd4e0965dac4700441295e3f"
        )

    def test_gray_float_uses_bt601_weights(self):
        rgba = np.array([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]], np.float32)
        gray = rgba_to_format(rgba, "gray")
        assert gray.dtype == np.float32
        np.testing.assert_allclose(gray, [[0.299, 0.587, 0.114]], rtol=1e-6)

    @pytest.mark.parametrize(
        ("format", "channels"),
        [("rgb", [0, 1, 2]), ("bgr", [2, 1, 0]), ("rgba", [0, 1, 2, 3]), ("bgra", [2, 1, 0, 3])],
    )
    def test_channel_orders(self, format, channels):
        rgba = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
        converted = rgba_to_format(rgba, format)
        np.testing.assert_array_equal(converted, rgba[..., channels])
        assert converted.flags.c_contiguous

    def test_unknown_format(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            rgba_to_format(np.zeros((1, 1, 4), np.uint8), "hsv")
