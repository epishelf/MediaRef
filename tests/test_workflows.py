"""Integration tests for MediaRef.

These tests verify end-to-end workflows and interactions between components.
"""

from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from mediaref import DataURI, MediaRef, batch_decode, cleanup_cache


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_image_workflow_file_to_embedding(self, sample_image_file: Path):
        """Test complete workflow: load image -> convert -> embed -> load again."""
        # 1. Create MediaRef from file
        ref = MediaRef(uri=str(sample_image_file))
        assert not ref.is_embedded
        assert not ref.is_remote

        # 2. Load as RGB array
        rgb = ref.to_ndarray()
        assert rgb.shape == (48, 64, 3)

        # 3. Create DataURI and MediaRef in one step
        embedded_ref = MediaRef(uri=DataURI.from_image(rgb, format="png"))
        assert embedded_ref.is_embedded
        assert not ref.is_embedded

        # 4. Load from embedded ref
        embedded_rgb = embedded_ref.to_ndarray()

        # 5. Verify data is identical (PNG is lossless)
        np.testing.assert_array_equal(rgb, embedded_rgb)

    @pytest.mark.tensorcodec
    def test_video_workflow_batch_processing(self, sample_video_file: tuple[Path, list[int]]):
        """Test complete workflow: create video refs -> batch decode -> process."""
        video_path, timestamps = sample_video_file

        # 1. Create MediaRefs for multiple frames (timestamps already in nanoseconds)
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps[:5]]

        # 2. Verify all are video references
        for ref in refs:
            assert ref.is_video
            assert not ref.is_remote
            assert not ref.is_embedded

        # 3. Batch decode
        frames = batch_decode(refs)
        assert len(frames) == 5

        # 4. Process frames (e.g., compute mean intensity)
        intensities = [frame.mean() for frame in frames]

        # 5. Verify frames are different
        assert len(set(intensities)) > 1  # At least some frames should differ

        # 6. Cleanup
        cleanup_cache()

    @pytest.mark.tensorcodec
    def test_mixed_media_workflow(self, sample_image_files: list[Path], sample_video_file: tuple[Path, list[int]]):
        """Test workflow with mixed images and videos."""
        video_path, timestamps = sample_video_file

        # 1. Create mixed MediaRefs (timestamps already in nanoseconds)
        refs = [
            MediaRef(uri=str(sample_image_files[0])),
            MediaRef(uri=str(video_path), pts_ns=timestamps[0]),
            MediaRef(uri=str(sample_image_files[1])),
            MediaRef(uri=str(video_path), pts_ns=timestamps[1]),
        ]

        # 2. Batch decode all
        results = batch_decode(refs, allow_images=True)
        assert len(results) == 4

        # 3. Verify all have same shape
        for rgb in results:
            assert rgb.shape == (48, 64, 3)

        # 4. Cleanup
        cleanup_cache()

    def test_serialization_workflow(self, sample_image_file: Path):
        """Test workflow: create -> serialize -> deserialize -> verify."""
        # 1. Create MediaRef
        original = MediaRef(uri=str(sample_image_file))

        # 2. Serialize to JSON
        json_str = original.model_dump_json()

        # 3. Deserialize
        restored = MediaRef.model_validate_json(json_str)

        # 4. Verify equality
        assert restored == original
        assert restored.uri == original.uri

        # 5. Verify functionality
        original_rgb = original.to_ndarray()
        restored_rgb = restored.to_ndarray()
        np.testing.assert_array_equal(original_rgb, restored_rgb)

    def test_path_resolution_workflow(self, tmp_path: Path):
        """Test workflow: create relative ref -> resolve -> load."""
        # 1. Create test image
        test_image = tmp_path / "images" / "test.png"
        test_image.parent.mkdir(parents=True)

        # Create a simple test image
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        img[:, :] = [0, 0, 255]  # Blue in RGB
        PIL.Image.fromarray(img).save(test_image)

        # 2. Create relative MediaRef
        relative_ref = MediaRef(uri="images/test.png")
        assert relative_ref.is_relative_path

        # 3. Resolve against base path
        resolved_ref = relative_ref.resolve_relative_path(str(tmp_path))
        assert not resolved_ref.is_relative_path
        assert tmp_path.as_posix() in resolved_ref.uri

        # 4. Load from resolved ref
        rgb = resolved_ref.to_ndarray()
        assert rgb.shape == (48, 64, 3)


@pytest.mark.integration
class TestDataURIIntegration:
    """Test data URI integration across components."""

    def test_data_uri_creation_and_loading(self, sample_image_file: Path):
        """Test creating and loading data URIs with different formats."""
        original_rgb = MediaRef(uri=str(sample_image_file)).to_ndarray()

        # Test PNG (lossless)
        png_ref = MediaRef(uri=DataURI.from_image(original_rgb, format="png"))
        np.testing.assert_array_equal(original_rgb, png_ref.to_ndarray())

        # Test JPEG (lossy)
        jpeg_ref = MediaRef(uri=DataURI.from_image(original_rgb, format="jpeg", quality=90))
        jpeg_rgb = jpeg_ref.to_ndarray()
        assert jpeg_rgb.shape == original_rgb.shape
        np.testing.assert_allclose(original_rgb, jpeg_rgb, atol=30)

    def test_data_uri_serialization(self, sample_data_uri: str):
        """Test serializing and deserializing data URIs."""
        # Create from data URI
        ref = MediaRef(uri=sample_data_uri)

        # Serialize
        json_str = ref.model_dump_json()

        # Deserialize
        restored = MediaRef.model_validate_json(json_str)

        # Verify
        assert restored.uri == ref.uri
        assert restored.is_embedded

        # Load and compare
        original_rgb = ref.to_ndarray()
        restored_rgb = restored.to_ndarray()
        np.testing.assert_array_equal(original_rgb, restored_rgb)

    @pytest.mark.tensorcodec
    def test_video_frame_to_data_uri(self, sample_video_file: tuple[Path, list[int]]):
        """Test converting video frame to data URI."""
        video_path, timestamps = sample_video_file

        # Load video frame and convert to DataURI
        original_rgb = MediaRef(uri=str(video_path), pts_ns=timestamps[1]).to_ndarray()
        embedded_ref = MediaRef(uri=DataURI.from_image(original_rgb, format="png"))

        # Verify roundtrip
        np.testing.assert_array_equal(original_rgb, embedded_ref.to_ndarray())


@pytest.mark.integration
@pytest.mark.network
class TestRemoteURLIntegration:
    """Test remote URL integration."""

    def test_remote_url_loading(self, remote_test_image_url: str):
        """Test loading from remote URL."""
        ref = MediaRef(uri=remote_test_image_url)

        assert ref.is_remote
        assert not ref.is_embedded

        # Load image
        rgb = ref.to_ndarray()
        assert isinstance(rgb, np.ndarray)
        assert len(rgb.shape) == 3
        assert rgb.shape[2] == 3

    def test_remote_url_to_pil(self, remote_test_image_url: str):
        """Test converting remote URL to PIL Image."""
        from PIL import Image

        ref = MediaRef(uri=remote_test_image_url)
        pil_img = ref.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"

    def test_remote_url_embedding(self, remote_test_image_url: str):
        """Test embedding remote URL as data URI."""
        original_rgb = MediaRef(uri=remote_test_image_url).to_ndarray()
        embedded_ref = MediaRef(uri=DataURI.from_image(original_rgb, format="png"))

        # Verify roundtrip
        np.testing.assert_array_equal(original_rgb, embedded_ref.to_ndarray())

    @pytest.mark.tensorcodec
    def test_remote_video_loading(self):
        """Test loading video frame from remote URL."""
        remote_video_url = (
            "https://raw.githubusercontent.com/open-world-agents/MediaRef/main/tests/assets/example_video.mkv"
        )
        pts_ns = 1_000_000_000

        ref = MediaRef(uri=remote_video_url, pts_ns=pts_ns)

        assert ref.is_remote
        assert ref.is_video
        assert not ref.is_embedded

        # Load video frame
        rgb = ref.to_ndarray()
        assert isinstance(rgb, np.ndarray)
        assert len(rgb.shape) == 3
        assert rgb.shape[2] == 3
        assert rgb.dtype == np.uint8


@pytest.mark.integration
class TestPropertyInteractions:
    """Test interactions between different properties."""

    def test_mutually_exclusive_properties(self, sample_image_file: Path, sample_data_uri: str):
        """Test that certain properties are mutually exclusive."""
        # Local file
        local_ref = MediaRef(uri=str(sample_image_file))
        assert not local_ref.is_remote
        assert not local_ref.is_embedded

        # Embedded
        embedded_ref = MediaRef(uri=sample_data_uri)
        assert embedded_ref.is_embedded
        assert not embedded_ref.is_remote

        # Remote
        remote_ref = MediaRef(uri="https://example.com/image.jpg")
        assert remote_ref.is_remote
        assert not remote_ref.is_embedded

    @pytest.mark.tensorcodec
    def test_video_property_combinations(self, sample_video_file: tuple[Path, list[int]]):
        """Test video property combinations."""
        video_path, timestamps = sample_video_file

        # Video frame (local file + video) - timestamps already in nanoseconds
        video_ref = MediaRef(uri=str(video_path), pts_ns=timestamps[0])
        assert video_ref.is_video
        assert not video_ref.is_remote
        assert not video_ref.is_embedded

        # Image from same file (local file but not video)
        image_ref = MediaRef(uri=str(video_path))
        assert not image_ref.is_video
        assert not image_ref.is_remote
        assert not image_ref.is_embedded


@pytest.mark.tensorcodec
class TestRGBADecoding:
    """Test RGBA decoding stability.

    Regression tests for RGBA format crash issue.
    PR: https://github.com/open-world-agents/MediaRef/pull/10
    Related: https://github.com/PyAV-Org/PyAV/issues/1269
    """

    # The specific video that caused crashes in v0.4.2-0.4.3
    # Originally from: https://huggingface.co/datasets/open-world-agents/example_dataset/resolve/main/example.mkv
    PROBLEMATIC_VIDEO_PATH = Path(__file__).parent / "assets" / "example.mkv"

    def test_rgba_format_no_crash(self):
        """Test that RGBA decoding doesn't crash on problematic videos.

        This is a regression test for the malloc_consolidate crash that occurred
        when using to_ndarray(format="rgba") on certain videos.

        Root cause: PyAV's to_rgb() with dst_format="rgba" produced ARGB instead
        of RGBA, causing memory corruption when the buffer was interpreted as RGBA.

        Fix: Decode using dst_format="argb" and manually reorder channels.
        """
        if not self.PROBLEMATIC_VIDEO_PATH.exists():
            pytest.skip("example.mkv not found")

        # This specific timestamp caused crashes in v0.4.2-0.4.3
        pts_ns = 1_000_000_000

        ref = MediaRef(uri=str(self.PROBLEMATIC_VIDEO_PATH), pts_ns=pts_ns)

        # Should not crash with malloc_consolidate error
        rgba = ref.to_ndarray(format="rgba")

        # Verify output is valid
        assert isinstance(rgba, np.ndarray)
        assert rgba.dtype == np.uint8
        assert len(rgba.shape) == 3
        assert rgba.shape[2] == 4  # RGBA channels

    def test_rgba_format_multiple_frames(self):
        """Test RGBA decoding stability across multiple frames."""
        if not self.PROBLEMATIC_VIDEO_PATH.exists():
            pytest.skip("example.mkv not found")

        # Test multiple timestamps including the problematic one
        timestamps_ns = [0, 500_000_000, 1_000_000_000, 1_500_000_000]

        refs = [MediaRef(uri=str(self.PROBLEMATIC_VIDEO_PATH), pts_ns=ts) for ts in timestamps_ns]

        # Batch decode should not crash
        results = batch_decode(refs)

        assert len(results) == len(timestamps_ns)
        for rgb in results:
            assert isinstance(rgb, np.ndarray)
            assert rgb.dtype == np.uint8
            assert rgb.shape[2] == 3


@pytest.mark.integration
class TestErrorRecovery:
    """Test error recovery and cleanup."""

    @pytest.mark.tensorcodec
    def test_cleanup_after_error(self, sample_video_file: tuple[Path, list[int]]):
        """Test that cleanup works after errors."""
        video_path, timestamps = sample_video_file

        # Load some frames (timestamps already in nanoseconds)
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps[:3]]
        batch_decode(refs)

        # Try to load invalid ref (should fail)
        invalid_refs = [MediaRef(uri="/nonexistent/video.mp4", pts_ns=0)]
        with pytest.raises(Exception):
            batch_decode(invalid_refs)

        # Cleanup should still work
        cleanup_cache()

        # Should be able to load again
        results = batch_decode(refs)
        assert len(results) == 3

    @pytest.mark.tensorcodec
    def test_multiple_cleanup_calls(self):
        """Test that multiple cleanup calls don't cause issues."""
        cleanup_cache()
        cleanup_cache()
        cleanup_cache()

    @pytest.mark.tensorcodec
    def test_load_after_cleanup(self, sample_video_file: tuple[Path, list[int]]):
        """Test loading after cleanup."""
        video_path, timestamps = sample_video_file

        ref = MediaRef(uri=str(video_path), pts_ns=timestamps[0])  # Already in nanoseconds

        # Load, cleanup, load again
        rgb1 = ref.to_ndarray()
        cleanup_cache()
        rgb2 = ref.to_ndarray()

        # Results should be identical
        np.testing.assert_array_equal(rgb1, rgb2)
