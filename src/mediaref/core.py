"""Core MediaRef class."""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Mapping, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
from pydantic import BaseModel, BeforeValidator, Field

from ._internal import is_cloud_uri as _is_cloud_uri

if TYPE_CHECKING:
    from .data_uri import DataURI


def _convert_datauri_to_str(v: Union[str, "DataURI"]) -> str:
    """Convert DataURI object to string if provided."""
    from .data_uri import DataURI  # noqa: E402

    if isinstance(v, DataURI):
        return v.uri
    return v


class MediaRef(BaseModel):
    """Media reference for images and video frames.

    Supports multiple URI schemes:
    - File paths: "/absolute/path" or "relative/path"
    - File URIs: "file:///path/to/media"
    - HTTP/HTTPS URLs: "https://example.com/image.jpg"
    - Data URIs: "data:image/png;base64,..."
    - Video frames: Any of the above with pts_ns set

    Examples:
        >>> # Image reference
        >>> ref = MediaRef(uri="image.png")
        >>> rgb = ref.to_ndarray(format="rgb")  # Default RGB format
        >>>
        >>> # Video frame reference
        >>> ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        >>> frame = ref.to_ndarray()
        >>>
        >>> # Remote URL
        >>> ref = MediaRef(uri="https://example.com/image.jpg")
        >>> pil_img = ref.to_pil_image()
        >>>
        >>> # Embedded data URI (from file or array)
        >>> from mediaref import DataURI
        >>> data_uri = DataURI.from_file("image.png")  # or DataURI.from_image(array)
        >>> ref = MediaRef(uri=data_uri)  # Can pass DataURI directly
    """

    uri: Annotated[str, BeforeValidator(_convert_datauri_to_str)] = Field(
        ...,
        description="URI (data:image/png;base64,... | file:///path | http[s]://...) or posix file path (/absolute/path | relative/path)",
    )
    pts_ns: Optional[int] = Field(
        default=None,
        description="Video frame timestamp in nanoseconds",
    )

    def __init__(self, *, uri: Union[str, "DataURI"], pts_ns: Optional[int] = None, **kwargs) -> None:
        """Initialize MediaRef with URI and optional timestamp.

        Args:
            uri: URI string or DataURI object
            pts_ns: Optional video frame timestamp in nanoseconds
            **kwargs: Additional fields (for internal use)
        """
        super().__init__(uri=uri, pts_ns=pts_ns, **kwargs)

    # ========== Properties ==========

    @property
    def is_embedded(self) -> bool:
        """True if this is embedded data (data URI)."""
        return self.uri.startswith("data:")

    @property
    def is_video(self) -> bool:
        """True if this references video media."""
        return self.pts_ns is not None

    @property
    def is_remote(self) -> bool:
        """True if this references a remote URL (http/https)."""
        return self.uri.startswith(("http://", "https://"))

    @property
    def is_cloud_uri(self) -> bool:
        """True if this URI is delegated to fsspec.

        Open-set: any scheme other than ``file`` / ``data`` (and not a bare
        path) is fsspec-routable. This includes ``http(s)://`` (handled by
        fsspec's HTTPFileSystem) and cloud schemes (``s3://``, ``gs://``,
        ``hf://``, …). fsspec is a core dependency; backends like
        ``s3fs``/``gcsfs``/``huggingface_hub`` must be installed for the
        schemes they serve. See [SPEC §2.1](../docs/SPEC.md) for guidance.

        Note: ``is_remote`` (http(s) specifically) and ``is_cloud_uri``
        overlap on http(s) URIs since the unified dispatch.
        """
        return _is_cloud_uri(self.uri)

    @property
    def is_relative_path(self) -> bool:
        """True if this is a relative path (not absolute, not URI).

        Local paths use platform-specific semantics; other URI schemes are
        delegated to fsspec.
        """
        if self.is_embedded or self.is_cloud_uri or self.uri.startswith("file://"):
            return False
        return not Path(self.uri).is_absolute()

    # ========== Path Utilities ==========

    def validate_uri(self, *, storage_options: Optional[Mapping[str, Any]] = None) -> bool:
        """Validate that the URI exists.

        Uses platform-specific path semantics (behavior differs on Windows vs POSIX).

        Returns:
            True if URI is valid/accessible

        Raises:
            ImportError: If the URI's fsspec backend is not installed.
        """
        if self.is_cloud_uri:
            from ._internal import cloud_uri_exists

            return cloud_uri_exists(self.uri, storage_options=storage_options)
        if self.is_embedded:
            return True  # Embedded data is always "valid"
        return Path(self.uri).exists()

    def resolve_relative_path(
        self,
        base_path: str,
        on_unresolvable: Literal["error", "warn", "ignore"] = "warn",
    ) -> "MediaRef":
        """Resolve relative path against a base path.

        Uses platform-specific path semantics (behavior differs on Windows vs POSIX).

        Args:
            base_path: Base path to resolve against
            on_unresolvable: How to handle unresolvable URIs (embedded/remote):
                - "error": Raise ValueError
                - "warn": Issue warning and return unchanged (default)
                - "ignore": Silently return unchanged

        Returns:
            New MediaRef with resolved absolute path

        Raises:
            ValueError: If URI is unresolvable and on_unresolvable="error"

        Examples:
            >>> ref = MediaRef(uri="relative/video.mkv", pts_ns=123456)
            >>> ref_resolved = ref.resolve_relative_path("/data/recordings")
            >>> # ref_resolved.uri == "/data/recordings/relative/video.mkv"
            >>>
            >>> # Handle unresolvable URIs
            >>> remote = MediaRef(uri="https://example.com/image.jpg")
            >>> remote.resolve_relative_path("/data/recordings", on_unresolvable="ignore")
        """
        if self.is_embedded or self.is_cloud_uri:
            kind = "embedded" if self.is_embedded else "cloud"
            if on_unresolvable == "error":
                raise ValueError(f"Cannot resolve unresolvable URI ({kind}): {self.uri}")
            elif on_unresolvable == "warn":
                warnings.warn(f"Cannot resolve unresolvable URI ({kind}): {self.uri}")
            return self

        if not self.is_relative_path:
            return self  # Already absolute or not a local path

        base_path_obj = Path(base_path)
        resolved_path = (base_path_obj / self.uri).as_posix()
        return MediaRef(uri=resolved_path, pts_ns=self.pts_ns)

    # ========== Loading Methods ==========

    def to_ndarray(
        self,
        format: Literal["rgb", "bgr", "rgba", "bgra", "gray", "native"] = "rgb",
        *,
        decoder: Literal["tensorcodec", "pyav", "torchcodec"] = "tensorcodec",
        decoder_options: Optional[Mapping[str, Any]] = None,
        image_decoder: Literal["tensorcodec", "torchcodec"] = "tensorcodec",
        image_decoder_options: Optional[Mapping[str, Any]] = None,
        storage_options: Optional[Mapping[str, Any]] = None,
    ) -> npt.NDArray[np.generic]:
        """Load and return media as numpy ndarray in specified format.

        Args:
            format: Output format (default: "rgb")
                - "rgb": RGB color (H, W, 3)
                - "bgr": BGR color (H, W, 3)
                - "rgba": RGB with alpha (H, W, 4)
                - "bgra": BGR with alpha (H, W, 4)
                - "gray": Grayscale (H, W)
                - "native": Preserve video sample values (PyAV or TensorCodec); grayscale
                  is (H, W), packed color is (H, W, C). No unit conversion.
            decoder: Video decoder backend. Ignored for image refs.
            decoder_options: Options passed to the video decoder constructor.
            image_decoder: Image decoder backend. Ignored for video refs.
            image_decoder_options: Options passed to the image backend's ``decode_image``.
            storage_options: Credentials and backend options passed to fsspec.
        Returns:
            Numpy ndarray in requested format

        Raises:
            ImportError: If video dependencies are not installed (for video frames)
            ValueError: If format is invalid

        Examples:
            >>> ref = MediaRef(uri="image.png")
            >>> rgb = ref.to_ndarray(format="rgb")  # Default RGB format
            >>>
            >>> ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
            >>> frame = ref.to_ndarray()  # Requires: pip install mediaref[video]
        """
        if format == "native":
            from .batch import batch_decode

            return batch_decode(
                [self],
                decoder=decoder,
                output_format="native",
                decoder_options=decoder_options,
                storage_options=storage_options,
            )[0]
        if decoder_options and decoder_options.get("output_format", "rgb") != "rgb":
            raise ValueError("Use format='native' to request native video output")
        rgba = self._load_as_rgba(
            decoder=decoder,
            decoder_options=decoder_options,
            image_decoder=image_decoder,
            image_decoder_options=image_decoder_options,
            storage_options=storage_options,
        )

        CONVERSION_MAP = {
            "rgb": cv2.COLOR_RGBA2RGB,
            "bgr": cv2.COLOR_RGBA2BGR,
            "bgra": cv2.COLOR_RGBA2BGRA,
            "gray": cv2.COLOR_RGBA2GRAY,
        }
        if format == "rgba":
            return rgba
        if format in CONVERSION_MAP:
            return cv2.cvtColor(rgba, CONVERSION_MAP[format])  # type: ignore[return-value]

        raise ValueError(f"Unsupported format: {format}. Must be one of: rgb, bgr, rgba, bgra, gray")

    def to_pil_image(
        self,
        format: Literal["rgb", "rgba", "gray"] = "rgb",
        *,
        decoder: Literal["tensorcodec", "pyav", "torchcodec"] = "tensorcodec",
        decoder_options: Optional[Mapping[str, Any]] = None,
        image_decoder: Literal["tensorcodec", "torchcodec"] = "tensorcodec",
        image_decoder_options: Optional[Mapping[str, Any]] = None,
        storage_options: Optional[Mapping[str, Any]] = None,
    ) -> PIL.Image.Image:
        """Load and return media as PIL Image.

        Args:
            format: Output format (default: "rgb")
                - "rgb": RGB color
                - "rgba": RGB with alpha
                - "gray": Grayscale
            decoder: Video decoder backend. Ignored for image refs.
            decoder_options: Options passed to the video decoder constructor.
            image_decoder: Image decoder backend. Ignored for video refs.
            image_decoder_options: Options passed to the image backend's ``decode_image``.
            storage_options: Credentials and backend options passed to fsspec.

        Returns:
            PIL Image object

        Raises:
            ImportError: If video dependencies are not installed (for video frames)
            ValueError: If format is invalid

        Examples:
            >>> ref = MediaRef(uri="image.png")
            >>> img = ref.to_pil_image()
        """
        if format in ("bgr", "bgra"):
            raise ValueError(f"Format '{format}' is not compatible with to_pil_image. Use 'rgb', 'rgba', or 'gray'.")

        array = self.to_ndarray(
            format=format,
            decoder=decoder,
            decoder_options=decoder_options,
            image_decoder=image_decoder,
            image_decoder_options=image_decoder_options,
            storage_options=storage_options,
        )
        if array.dtype != np.uint8:
            raise ValueError(
                f"to_pil_image() requires uint8 output, got {array.dtype}. "
                "Use to_ndarray() to preserve high-dynamic-range data."
            )
        return PIL.Image.fromarray(array)

    # ========== Internal ==========

    def _load_as_rgba(
        self,
        *,
        decoder: Literal["tensorcodec", "pyav", "torchcodec"] = "tensorcodec",
        decoder_options: Optional[Mapping[str, Any]] = None,
        image_decoder: Literal["tensorcodec", "torchcodec"] = "tensorcodec",
        image_decoder_options: Optional[Mapping[str, Any]] = None,
        storage_options: Optional[Mapping[str, Any]] = None,
    ) -> npt.NDArray[np.generic]:
        """Internal: Load media as RGBA array.

        Raises:
            ImportError: If video dependencies are not installed (for video frames)
        """
        from ._internal import load_image_as_rgba, load_video_frame_as_rgba

        if self.is_video:
            assert self.pts_ns is not None  # Type guard: is_video ensures pts_ns is not None
            return load_video_frame_as_rgba(
                self.uri,
                self.pts_ns,
                decoder=decoder,
                decoder_options=decoder_options,
                storage_options=storage_options,
            )
        else:
            return load_image_as_rgba(
                self.uri,
                decoder=image_decoder,
                decoder_options=image_decoder_options,
                storage_options=storage_options,
            )
