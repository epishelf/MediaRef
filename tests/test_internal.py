"""Tests for internal utility functions.

These tests verify internal implementation details not exposed by the public API:
- Internal RGBA format handling (load_image_as_rgba)
"""

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from mediaref._internal import load_image_as_rgba
from mediaref.data_uri import DataURI


class TestInternalRGBAHandling:
    """Test internal RGBA format handling.

    These tests verify that load_image_as_rgba() correctly handles different
    image formats and maintains data integrity through encode/decode cycles.
    """

    def test_load_png_as_rgba_lossless(self, sample_rgba_array: npt.NDArray[np.uint8]):
        """Test that PNG encoding/decoding via load_image_as_rgba is lossless."""
        # Create data URI from RGBA array
        rgb_array = cv2.cvtColor(sample_rgba_array, cv2.COLOR_RGBA2RGB)
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
        rgb_array = cv2.cvtColor(sample_rgba_array, cv2.COLOR_RGBA2RGB)
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
        rgb_array = cv2.cvtColor(sample_rgba_array, cv2.COLOR_RGBA2RGB)
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
