"""Tests for MediaRef loading methods (to_ndarray, to_pil_image) and DataURI.

Video cases require the TensorCodec backend.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from mediaref import DataURI, MediaRef


class TestToNdarrayImage:
    """Test to_ndarray method for images."""

    def test_to_ndarray_from_file(self, sample_image_file: Path):
        """Test loading RGB array from image file."""
        ref = MediaRef(uri=str(sample_image_file))
        rgb = ref.to_ndarray()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8

    def test_to_ndarray_color_correctness(self, sample_image_file: Path):
        """Test that RGB array has correct color values."""
        ref = MediaRef(uri=str(sample_image_file))
        rgb = ref.to_ndarray()

        # sample_image_file has BGR = (255, 0, 0) which is blue in BGR format
        # When converted to RGB, it should be (0, 0, 255) - blue in RGB format
        assert rgb[0, 0, 0] == 0  # Red channel
        assert rgb[0, 0, 1] == 0  # Green channel
        assert rgb[0, 0, 2] == 255  # Blue channel

    def test_to_ndarray_from_data_uri(self, sample_data_uri: str):
        """Test loading RGB array from data URI."""
        ref = MediaRef(uri=sample_data_uri)
        rgb = ref.to_ndarray()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8

    @pytest.mark.network
    def test_to_ndarray_from_remote_url(self, remote_test_image_url: str):
        """Test loading RGB array from remote URL."""
        ref = MediaRef(uri=remote_test_image_url)
        rgb = ref.to_ndarray()

        assert isinstance(rgb, np.ndarray)
        assert len(rgb.shape) == 3
        assert rgb.shape[2] == 3
        assert rgb.dtype == np.uint8

    def test_to_ndarray_from_file_uri(self, sample_image_file: Path):
        """Test loading RGB array from file:// URI."""
        file_uri = f"file://{sample_image_file.as_posix()}"
        ref = MediaRef(uri=file_uri)
        rgb = ref.to_ndarray()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8

    def test_to_ndarray_nonexistent_file(self):
        """Test that loading from nonexistent file raises error."""
        ref = MediaRef(uri="/nonexistent/file.png")

        with pytest.raises(Exception):  # Should raise ValueError or FileNotFoundError
            ref.to_ndarray()


@pytest.mark.tensorcodec
class TestToNdarrayVideo:
    """Test to_ndarray method for video frames."""

    def test_to_ndarray_from_video(self, sample_video_file: tuple[Path, list[int]]):
        """Test loading RGB array from video frame."""
        video_path, timestamps = sample_video_file
        pts_ns = timestamps[1]  # Second frame (already in nanoseconds)

        ref = MediaRef(uri=str(video_path), pts_ns=pts_ns)
        rgb = ref.to_ndarray()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8

    def test_to_ndarray_video_first_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test loading first frame from video."""
        video_path, timestamps = sample_video_file

        ref = MediaRef(uri=str(video_path), pts_ns=0)
        rgb = ref.to_ndarray()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)

    def test_to_ndarray_video_different_frames(self, sample_video_file: tuple[Path, list[int]]):
        """Test that different frames have different content."""
        video_path, timestamps = sample_video_file

        ref1 = MediaRef(uri=str(video_path), pts_ns=timestamps[0])
        ref2 = MediaRef(uri=str(video_path), pts_ns=timestamps[2])

        rgb1 = ref1.to_ndarray()
        rgb2 = ref2.to_ndarray()

        # Frames should be different (different intensities)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(rgb1, rgb2)

    def test_to_ndarray_video_nonexistent_file(self):
        """Test that loading from nonexistent video raises error."""
        ref = MediaRef(uri="/nonexistent/video.mp4", pts_ns=0)

        with pytest.raises(Exception):  # Should raise FileNotFoundError
            ref.to_ndarray()

    def test_to_ndarray_from_video_file_uri(self, sample_video_file: tuple[Path, list[int]]):
        """Test loading RGB array from video frame using file:// URI."""
        video_path, timestamps = sample_video_file
        file_uri = f"file://{video_path.as_posix()}"
        pts_ns = timestamps[1]

        ref = MediaRef(uri=file_uri, pts_ns=pts_ns)
        rgb = ref.to_ndarray()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8

    def test_to_ndarray_video_without_pts_ns_raises_error(self, sample_video_file: tuple[Path, list[int]]):
        """Test that loading video file without pts_ns raises ValueError.

        When pts_ns is not provided, MediaRef treats the file as an image.
        Trying to load a video file as an image fails with ValueError.
        """
        video_path, _ = sample_video_file
        ref = MediaRef(uri=str(video_path))  # No pts_ns, so is_video=False

        # Should raise ValueError because it tries to load video file as image
        with pytest.raises(ValueError, match="Failed to load image"):
            ref.to_ndarray()


class TestToPilImage:
    """Test to_pil_image method."""

    def test_to_pil_image_from_file(self, sample_image_file: Path):
        """Test loading PIL Image from file."""
        ref = MediaRef(uri=str(sample_image_file))
        pil_img = ref.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (64, 48)  # PIL uses (width, height)
        assert pil_img.mode == "RGB"

    def test_to_pil_image_from_data_uri(self, sample_data_uri: str):
        """Test loading PIL Image from data URI."""
        ref = MediaRef(uri=sample_data_uri)
        pil_img = ref.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (64, 48)
        assert pil_img.mode == "RGB"

    @pytest.mark.tensorcodec
    def test_to_pil_image_from_video(self, sample_video_file: tuple[Path, list[int]]):
        """Test loading PIL Image from video frame."""
        video_path, timestamps = sample_video_file
        pts_ns = timestamps[1]  # Already in nanoseconds

        ref = MediaRef(uri=str(video_path), pts_ns=pts_ns)
        pil_img = ref.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (64, 48)
        assert pil_img.mode == "RGB"

    def test_to_pil_image_matches_to_ndarray(self, sample_image_file: Path):
        """Test that to_pil_image matches to_ndarray."""
        ref = MediaRef(uri=str(sample_image_file))

        rgb_array = ref.to_ndarray()
        pil_img = ref.to_pil_image()
        pil_array = np.array(pil_img)

        np.testing.assert_array_equal(rgb_array, pil_array)

    @pytest.mark.network
    def test_to_pil_image_from_remote_url(self, remote_test_image_url: str):
        """Test loading PIL Image from remote URL."""
        ref = MediaRef(uri=remote_test_image_url)
        pil_img = ref.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"

    def test_to_pil_image_from_file_uri(self, sample_image_file: Path):
        """Test loading PIL Image from file:// URI."""
        file_uri = f"file://{sample_image_file.as_posix()}"
        ref = MediaRef(uri=file_uri)
        pil_img = ref.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (64, 48)
        assert pil_img.mode == "RGB"


class TestDataURICreation:
    """Test DataURI creation from various sources."""

    def test_from_file(self, sample_image_file: Path):
        """Test creating DataURI from file."""
        data_uri = DataURI.from_file(sample_image_file)

        assert data_uri.mimetype == "image/png"
        assert data_uri.is_base64
        assert data_uri.is_image
        assert data_uri.uri.startswith("data:image/png;base64,")
        assert len(data_uri) > 0

    def test_from_image_numpy_array(self, sample_image_file: Path):
        """Test creating DataURI from numpy array."""
        rgb = MediaRef(uri=str(sample_image_file)).to_ndarray()
        data_uri = DataURI.from_image(rgb, format="png")

        assert data_uri.mimetype == "image/png"
        assert data_uri.is_base64
        assert data_uri.uri.startswith("data:image/png;base64,")

    def test_from_image_pil_image(self, sample_image_file: Path):
        """Test creating DataURI from PIL Image."""
        pil_img = MediaRef(uri=str(sample_image_file)).to_pil_image()
        data_uri = DataURI.from_image(pil_img, format="png")

        assert data_uri.mimetype == "image/png"
        assert data_uri.is_base64
        assert data_uri.uri.startswith("data:image/png;base64,")

    def test_from_uri_string(self, sample_image_file: Path):
        """Test parsing DataURI from string."""
        original = DataURI.from_file(sample_image_file)
        parsed = DataURI.from_uri(original.uri)

        assert parsed.mimetype == original.mimetype
        assert parsed.is_base64 == original.is_base64
        assert parsed.data == original.data

    @pytest.mark.tensorcodec
    def test_from_video_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test creating DataURI from video frame."""
        video_path, timestamps = sample_video_file
        rgb = MediaRef(uri=str(video_path), pts_ns=timestamps[1]).to_ndarray()
        data_uri = DataURI.from_image(rgb, format="png")

        assert data_uri.mimetype == "image/png"
        assert data_uri.uri.startswith("data:image/png;base64,")

    def test_data_stored_as_is_base64(self, sample_image_file: Path):
        """Test that base64 data is stored as-is (base64 string as bytes)."""
        import base64

        data_uri = DataURI.from_file(sample_image_file)

        # data field should store base64 string as bytes
        assert isinstance(data_uri.data, bytes)
        assert data_uri.is_base64

        # data should be the base64 string (as bytes), not decoded
        base64_str = data_uri.data.decode("utf-8")
        # Should be valid base64
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0

        # uri should contain the same base64 string
        assert base64_str in data_uri.uri

    def test_data_stored_as_is_non_base64(self):
        """Test that non-base64 data is stored as-is (URL-encoded text as bytes)."""
        # Create a non-base64 data URI with URL-encoded text
        uri_str = "data:text/plain,Hello%20World"
        data_uri = DataURI.from_uri(uri_str)

        # data field should store URL-encoded text as bytes
        assert isinstance(data_uri.data, bytes)
        assert not data_uri.is_base64
        assert data_uri.data == b"Hello%20World"

        # uri should reconstruct correctly
        assert data_uri.uri == uri_str

    def test_quote_validation_accepts_quoted_data(self):
        """Test that properly URL-encoded data is accepted."""
        # Properly quoted data should be accepted
        uri_str = "data:text/plain,Hello%20World%21"
        data_uri = DataURI.from_uri(uri_str)

        assert data_uri.mimetype == "text/plain"
        assert not data_uri.is_base64
        assert data_uri.uri == uri_str

    def test_quote_validation_rejects_unquoted_data(self):
        """Test that unquoted data is rejected."""
        from pydantic import ValidationError

        # Unquoted space should be rejected
        with pytest.raises(ValidationError, match="unquoted characters"):
            DataURI(mimetype="text/plain", is_base64=False, data=b"Hello World")

        # Unquoted newline should be rejected
        with pytest.raises(ValidationError, match="unquoted characters"):
            DataURI(mimetype="text/plain", is_base64=False, data=b"Hello\nWorld")

        # Unquoted special characters should be rejected
        with pytest.raises(ValidationError, match="unquoted characters"):
            DataURI(mimetype="text/plain", is_base64=False, data=b"Hello&World")

    def test_base64_data_does_not_need_url_encoding(self):
        """Test that base64 data is accepted without URL encoding validation."""
        import base64

        # Base64 data can contain characters that would need quoting in non-base64
        # This should be accepted because is_base64=True
        base64_data = base64.b64encode(b"Hello World!").decode("utf-8")
        data_uri = DataURI(mimetype="text/plain", is_base64=True, data=base64_data.encode("utf-8"))

        assert data_uri.is_base64
        assert data_uri.decoded_data == b"Hello World!"


class TestDataURIFormats:
    """Test different image formats for DataURI."""

    @pytest.fixture
    def sample_rgb(self, sample_image_file: Path) -> npt.NDArray[np.uint8]:
        """Provide sample RGB array."""
        return MediaRef(uri=str(sample_image_file)).to_ndarray()

    def test_format_png(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test PNG format."""
        data_uri = DataURI.from_image(sample_rgb, format="png")
        assert data_uri.mimetype == "image/png"
        assert data_uri.uri.startswith("data:image/png;base64,")

    def test_format_jpeg(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test JPEG format."""
        data_uri = DataURI.from_image(sample_rgb, format="jpeg")
        assert data_uri.mimetype == "image/jpeg"
        assert data_uri.uri.startswith("data:image/jpeg;base64,")

    def test_format_bmp(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test BMP format."""
        data_uri = DataURI.from_image(sample_rgb, format="bmp")
        assert data_uri.mimetype == "image/bmp"
        assert data_uri.uri.startswith("data:image/bmp;base64,")

    def test_jpeg_quality_parameter(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test JPEG quality affects output size."""
        high_quality = DataURI.from_image(sample_rgb, format="jpeg", quality=95)
        low_quality = DataURI.from_image(sample_rgb, format="jpeg", quality=10)

        assert len(high_quality) > len(low_quality)

    def test_input_format_rgb(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test input_format='rgb' (default)."""
        data_uri = DataURI.from_image(sample_rgb, format="png", input_format="rgb")
        restored = data_uri.to_ndarray()

        # Should be lossless for PNG
        np.testing.assert_array_equal(sample_rgb, restored)

    def test_input_format_bgr(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test input_format='bgr' correctly converts BGR to RGB."""
        import cv2

        # Convert RGB to BGR
        bgr_array = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2BGR)

        # Create DataURI with input_format='bgr'
        data_uri = DataURI.from_image(bgr_array, format="png", input_format="bgr")
        restored = data_uri.to_ndarray()

        # Should match original RGB
        np.testing.assert_array_equal(sample_rgb, restored)

    def test_input_format_rgba(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test input_format='rgba' with 4-channel RGBA array."""
        import cv2

        # Convert RGB to RGBA
        rgba_array = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2RGBA)

        # Create DataURI with input_format='rgba'
        data_uri = DataURI.from_image(rgba_array, format="png", input_format="rgba")
        restored = data_uri.to_ndarray()

        # Should match original RGB
        np.testing.assert_array_equal(sample_rgb, restored)

    def test_input_format_bgra(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test input_format='bgra' with 4-channel BGRA array."""
        import cv2

        # Convert RGB to BGRA
        bgra_array = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2BGRA)

        # Create DataURI with input_format='bgra'
        data_uri = DataURI.from_image(bgra_array, format="png", input_format="bgra")
        restored = data_uri.to_ndarray()

        # Should match original RGB
        np.testing.assert_array_equal(sample_rgb, restored)

    def test_input_format_invalid_3channel(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test that invalid input_format for 3-channel array raises ValueError."""
        with pytest.raises(ValueError, match="Invalid input_format.*for 3-channel array"):
            DataURI.from_image(sample_rgb, format="png", input_format="rgba")  # type: ignore

    def test_input_format_invalid_4channel(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test that invalid input_format for 4-channel array raises ValueError."""
        import cv2

        rgba_array = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2RGBA)
        with pytest.raises(ValueError, match="Invalid input_format.*for 4-channel array"):
            DataURI.from_image(rgba_array, format="png", input_format="invalid")  # type: ignore


class TestDataURIConversion:
    """Test DataURI conversion methods."""

    @pytest.fixture
    def sample_data_uri(self, sample_image_file: Path) -> DataURI:
        """Provide sample DataURI."""
        return DataURI.from_file(sample_image_file)

    def test_to_ndarray(self, sample_data_uri: DataURI):
        """Test converting DataURI to RGB array."""
        rgb = sample_data_uri.to_ndarray()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8

    def test_to_pil_image(self, sample_data_uri: DataURI):
        """Test converting DataURI to PIL Image."""
        pil_img = sample_data_uri.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (64, 48)  # PIL uses (width, height)
        assert pil_img.mode == "RGB"

    def test_str_conversion(self, sample_data_uri: DataURI):
        """Test __str__ returns URI string."""
        uri_str = str(sample_data_uri)

        assert uri_str == sample_data_uri.uri
        assert uri_str.startswith("data:image/png;base64,")


class TestDataURIRoundtrip:
    """Test DataURI encoding/decoding roundtrips."""

    @pytest.fixture
    def sample_rgb(self, sample_image_file: Path) -> npt.NDArray[np.uint8]:
        """Provide sample RGB array."""
        return MediaRef(uri=str(sample_image_file)).to_ndarray()

    def test_png_lossless_roundtrip(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test PNG encoding is lossless."""
        data_uri = DataURI.from_image(sample_rgb, format="png")
        restored_rgb = data_uri.to_ndarray()

        np.testing.assert_array_equal(sample_rgb, restored_rgb)

    def test_bmp_lossless_roundtrip(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test BMP encoding is lossless."""
        data_uri = DataURI.from_image(sample_rgb, format="bmp")
        restored_rgb = data_uri.to_ndarray()

        np.testing.assert_array_equal(sample_rgb, restored_rgb)

    def test_jpeg_lossy_roundtrip(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test JPEG encoding is lossy but close."""
        data_uri = DataURI.from_image(sample_rgb, format="jpeg", quality=85)
        restored_rgb = data_uri.to_ndarray()

        assert sample_rgb.shape == restored_rgb.shape
        np.testing.assert_allclose(sample_rgb, restored_rgb, atol=30)

    def test_uri_string_roundtrip(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test parsing DataURI from string preserves data."""
        original = DataURI.from_image(sample_rgb, format="png")
        parsed = DataURI.from_uri(original.uri)

        assert parsed.mimetype == original.mimetype
        assert parsed.is_base64 == original.is_base64
        np.testing.assert_array_equal(parsed.to_ndarray(), original.to_ndarray())

    def test_rgba_alpha_preservation_png(self):
        """Test that alpha channel is preserved in PNG format."""
        # Create RGBA array with varying alpha values
        h, w = 10, 10
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = 100  # R
        rgba[:, :, 1] = 150  # G
        rgba[:, :, 2] = 200  # B
        # Create varying alpha values
        alpha_values = np.linspace(0, 255, h * w, dtype=np.uint8).reshape(h, w)
        rgba[:, :, 3] = alpha_values

        # Encode to PNG and decode
        data_uri = DataURI.from_image(rgba, format="png")
        restored_rgba = data_uri.to_ndarray(format="rgba")

        # Verify alpha channel is preserved
        np.testing.assert_array_equal(rgba, restored_rgba)

    def test_rgba_alpha_dropped_jpeg(self):
        """Test that alpha channel is dropped in JPEG format."""
        # Create RGBA array with varying alpha values
        h, w = 10, 10
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = 100  # R
        rgba[:, :, 1] = 150  # G
        rgba[:, :, 2] = 200  # B
        rgba[:, :, 3] = 128  # Alpha (should be dropped)

        # Encode to JPEG
        data_uri = DataURI.from_image(rgba, format="jpeg")

        # Decode as RGBA - alpha should be 255 (opaque)
        restored_rgba = data_uri.to_ndarray(format="rgba")

        # Verify alpha is 255 (opaque) everywhere
        assert np.all(restored_rgba[:, :, 3] == 255)

        # Verify RGB channels are approximately preserved (JPEG is lossy)
        np.testing.assert_allclose(rgba[:, :, :3], restored_rgba[:, :, :3], atol=30)


class TestDataURIWithMediaRef:
    """Test DataURI integration with MediaRef."""

    @pytest.fixture
    def sample_rgb(self, sample_image_file: Path) -> npt.NDArray[np.uint8]:
        """Provide sample RGB array."""
        return MediaRef(uri=str(sample_image_file)).to_ndarray()

    def test_mediaref_accepts_datauri_object(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test MediaRef accepts DataURI object directly."""
        data_uri = DataURI.from_image(sample_rgb, format="png")
        ref = MediaRef(uri=data_uri)  # type: ignore[arg-type]

        assert ref.is_embedded
        np.testing.assert_array_equal(ref.to_ndarray(), sample_rgb)

    def test_mediaref_accepts_datauri_string(self, sample_rgb: npt.NDArray[np.uint8]):
        """Test MediaRef accepts DataURI string."""
        data_uri = DataURI.from_image(sample_rgb, format="png")
        ref = MediaRef(uri=str(data_uri))

        assert ref.is_embedded
        np.testing.assert_array_equal(ref.to_ndarray(), sample_rgb)

    @pytest.mark.tensorcodec
    def test_video_frame_to_datauri(self, sample_video_file: tuple[Path, list[int]]):
        """Test creating DataURI from video frame."""
        video_path, timestamps = sample_video_file
        original_rgb = MediaRef(uri=str(video_path), pts_ns=timestamps[1]).to_ndarray()

        data_uri = DataURI.from_image(original_rgb, format="png")
        ref = MediaRef(uri=data_uri)  # type: ignore[arg-type]

        assert ref.is_embedded
        np.testing.assert_array_equal(ref.to_ndarray(), original_rgb)


class TestLoadingErrorHandling:
    """Test error handling in loading methods."""

    def test_load_corrupted_image_file(self, tmp_path: Path):
        """Test loading corrupted image file raises error."""
        corrupted_file = tmp_path / "corrupted.png"
        corrupted_file.write_bytes(b"not an image")

        ref = MediaRef(uri=str(corrupted_file))

        with pytest.raises(Exception):  # Should raise ValueError
            ref.to_ndarray()

    def test_load_invalid_data_uri(self):
        """Test loading invalid data URI raises error."""
        ref = MediaRef(uri="data:image/png;base64,invalid_base64!")

        with pytest.raises(Exception):  # Should raise ValueError
            ref.to_ndarray()

    def test_load_unsupported_format(self, sample_image_file: Path):
        """Test that unsupported format in DataURI raises error."""
        ref = MediaRef(uri=str(sample_image_file))
        rgb = ref.to_ndarray()

        with pytest.raises(Exception):  # Should raise ValueError
            DataURI.from_image(rgb, format="invalid")  # type: ignore
