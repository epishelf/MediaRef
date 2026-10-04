"""Internal loading and encoding utilities."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.request import url2pathname

import fsspec
import numpy as np
import numpy.typing as npt
import PIL.Image

NANOSECOND = 1_000_000_000  # 1 second in nanoseconds

# URI schemes MediaRef handles directly. Anything else with a scheme is
# delegated to fsspec.open(), which dispatches to the appropriate backend
# (HTTPFileSystem, S3FileSystem, GCSFileSystem, HfFileSystem, …). All
# backends return seekable file-likes that map seek to HTTP Range requests,
# enabling sparse video frame access without downloading the full file.
_DIRECT_URI_SCHEMES = frozenset({"file", "data"})
_FILE_URI_PREFIX = "file://"


def _scheme_of(uri: str) -> str:
    """Lowercased URI scheme (the part before ``://``), or ``""`` for paths."""
    scheme, sep, _ = uri.partition("://")
    return scheme.lower() if sep else ""


def is_cloud_uri(uri: str) -> bool:
    """True if ``uri`` is delegated to fsspec.

    Open-set: any URI whose scheme is not in :data:`_DIRECT_URI_SCHEMES`
    (``file``, ``data``) and is not a bare path is treated as fsspec-routable.
    This includes ``http(s)://`` (handled by fsspec's HTTPFileSystem +
    aiohttp backend) and any cloud scheme whose backend is installed
    (``s3fs`` for ``s3://``, ``gcsfs`` for ``gs://``, ``huggingface_hub``
    for ``hf://``, …). Backend missing → fsspec raises a clear error.
    """
    scheme = _scheme_of(uri)
    return bool(scheme) and scheme not in _DIRECT_URI_SCHEMES


def _freeze_cache_value(value: Any) -> Any:
    """Return a deterministic, repr-safe shape for cache-key hashing."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_cache_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_cache_value(item) for item in value), key=repr))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return ("bytes", len(value), hashlib.sha256(value).hexdigest())
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return (
            "ndarray",
            contiguous.dtype.str,
            contiguous.shape,
            hashlib.sha256(contiguous.tobytes()).hexdigest(),
        )
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return (type(value).__module__, type(value).__qualname__, id(value))


def make_cache_key(source: str, *options: Mapping[str, Any]) -> str:
    """Build an opaque cache key from a source and option mappings."""
    nonempty = tuple(option for option in options if option)
    if not nonempty:
        return source
    payload = _freeze_cache_value(nonempty)
    digest = hashlib.sha256(repr(payload).encode()).hexdigest()
    return f"{source}#{digest}"


def open_cloud(uri: str, storage_options: Optional[Mapping[str, Any]] = None):
    """Open a fsspec-routed URI as a binary file-like (context manager)."""
    return fsspec.open(uri, "rb", **dict(storage_options or {}))


def cloud_uri_exists(uri: str, storage_options: Optional[Mapping[str, Any]] = None) -> bool:
    """Return whether a fsspec-routed URI exists."""
    filesystem, path = fsspec.core.url_to_fs(uri, **dict(storage_options or {}))
    return filesystem.exists(path)


def _file_uri_to_path(uri: str) -> str:
    """Convert a ``file://`` URI to a local filesystem path (handles URL decoding)."""
    return url2pathname(uri[len(_FILE_URI_PREFIX) :])


def _resolve_to_local_path(uri: str) -> str:
    """Resolve a ``file://`` URI or bare path to an existing local path.

    Raises:
        FileNotFoundError: If the resolved path doesn't exist.
    """
    path = _file_uri_to_path(uri) if uri.startswith(_FILE_URI_PREFIX) else uri
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def resolve_video_source(uri: str) -> str:
    """Return the source string the video decoder layer can consume directly.

    For cloud URIs (``hf://``, ``s3://``, …) the URI is returned as-is — the
    decoder layer (``cached_av``) opens fsspec internally and ties the
    file-like's lifetime to its cache entry. For ``file://`` URIs and bare
    paths, returns the verified local path string.

    Raises:
        FileNotFoundError: For local paths that don't exist.
    """
    if is_cloud_uri(uri):
        return uri
    return _resolve_to_local_path(uri)


# ============================================================================
# Image Loading
# ============================================================================


def load_image_as_rgba(
    path_or_uri: str,
    *,
    decoder: str = "tensorcodec",
    decoder_options: Optional[Mapping[str, Any]] = None,
    storage_options: Optional[Mapping[str, Any]] = None,
) -> npt.NDArray[np.generic]:
    """Load image from any source and return as RGBA numpy array.

    Args:
        path_or_uri: File path, URL, or data URI.
        decoder: Image decoder backend (``"tensorcodec"`` or ``"torchcodec"``).
        decoder_options: Options passed to the backend's ``decode_image``.
        storage_options: Credentials and backend options passed to fsspec.

    Returns:
        RGBA numpy array

    Raises:
        ValueError: If loading fails
        FileNotFoundError: If local file doesn't exist
    """
    try:
        if decoder not in ("tensorcodec", "torchcodec"):
            raise ValueError(f"Unknown image decoder backend: {decoder}. Must be 'tensorcodec' or 'torchcodec'")
        if path_or_uri.startswith("data:"):
            from .data_uri import DataURI

            source: str | bytes = DataURI.from_uri(path_or_uri).decoded_data
        elif is_cloud_uri(path_or_uri):
            with open_cloud(path_or_uri, storage_options=storage_options) as f:
                source = f.read()
        else:
            source = _resolve_to_local_path(path_or_uri)
        if decoder == "torchcodec":
            return _decode_torchcodec_image_as_rgba(source, decoder_options)
        return decode_image_as_rgba(source, decoder_options)
    except FileNotFoundError:
        raise
    except ImportError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load image from {path_or_uri}: {e}") from e


def _image_options(decoder_options: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    options = dict(decoder_options or {})
    if "mode" in options:
        raise ValueError("MediaRef controls the image decoder's mode; use to_ndarray(format=...) instead")
    return options


def _chw_to_rgba(array: npt.NDArray[np.generic], backend: str) -> npt.NDArray[np.generic]:
    """Normalize a CHW image (or NCHW animation; first frame) to RGBA HWC."""
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[0] != 4:
        raise ValueError(f"{backend} returned an unexpected image shape: {array.shape}")
    return np.ascontiguousarray(np.moveaxis(array, 0, -1))


def decode_image_as_rgba(
    source: str | bytes,
    decoder_options: Optional[Mapping[str, Any]] = None,
) -> npt.NDArray[np.generic]:
    """Decode an encoded image (path or bytes) with TensorCodec to RGBA HWC.

    TensorCodec applies EXIF (JPEG/PNG/WebP) and AVIF orientation.
    """
    from tensorcodec.decoders import decode_image

    return _chw_to_rgba(decode_image(source, mode="RGB_ALPHA", **_image_options(decoder_options)), "TensorCodec")


def _decode_torchcodec_image_as_rgba(
    source: str | bytes,
    decoder_options: Optional[Mapping[str, Any]],
) -> npt.NDArray[np.generic]:
    """Decode an image through TorchCodec 0.16+ and normalize CHW to RGBA HWC."""
    try:
        from torchcodec.decoders import decode_image
    except (ImportError, RuntimeError) as e:
        raise ImportError(
            "TorchCodec image decoding requires Python>=3.10 and torchcodec>=0.16. "
            "Install with: pip install 'mediaref[torchcodec]'"
        ) from e

    tensor = decode_image(source, mode="RGB_ALPHA", **_image_options(decoder_options))
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    return _apply_exif_orientation(_chw_to_rgba(array, "TorchCodec"), source)


def _apply_exif_orientation(
    array: npt.NDArray[np.generic],
    source: str | bytes,
) -> npt.NDArray[np.generic]:
    """Match Pillow's EXIF orientation behavior without narrowing the dtype."""
    pil_source: str | io.BytesIO = source if isinstance(source, str) else io.BytesIO(source)
    try:
        with PIL.Image.open(pil_source) as image:
            orientation = image.getexif().get(274, 1)
    except OSError:
        # TorchCodec supports formats (notably AVIF/HEIC) that a given Pillow
        # build may not recognize. EXIF normalization must not narrow support.
        return np.ascontiguousarray(array)

    if orientation == 2:
        array = np.flip(array, axis=1)
    elif orientation == 3:
        array = np.rot90(array, 2)
    elif orientation == 4:
        array = np.flip(array, axis=0)
    elif orientation == 5:
        array = np.transpose(array, (1, 0, 2))
    elif orientation == 6:
        array = np.rot90(array, 3)
    elif orientation == 7:
        array = np.flip(np.transpose(array, (1, 0, 2)), axis=(0, 1))
    elif orientation == 8:
        array = np.rot90(array, 1)
    return np.ascontiguousarray(array)


# ============================================================================
# Video Loading
# ============================================================================


def load_video_frame_as_rgba(
    path_or_url: str,
    pts_ns: int,
    *,
    decoder: str = "tensorcodec",
    decoder_options: Optional[Mapping[str, Any]] = None,
    storage_options: Optional[Mapping[str, Any]] = None,
) -> npt.NDArray[np.generic]:
    """Load video frame and return as RGBA numpy array.

    Args:
        path_or_url: File path or URL to video
        pts_ns: Presentation timestamp in nanoseconds

    Returns:
        RGBA numpy array

    Raises:
        ImportError: If video dependencies are not installed
        ValueError: If loading fails
        FileNotFoundError: If local file doesn't exist
    """
    from .batch import _get_decoder_class

    decoder_class = _get_decoder_class(decoder)

    pts_seconds = pts_ns / NANOSECOND

    try:
        source = resolve_video_source(path_or_url)
        options = dict(decoder_options or {})
        options["storage_options"] = storage_options
        with decoder_class(source, **options) as video_decoder:
            batch = video_decoder.get_frames_played_at([pts_seconds])
            rgb_nchw = batch.data[0]
            rgb_hwc = np.transpose(rgb_nchw, (1, 2, 0))
            if np.issubdtype(rgb_hwc.dtype, np.integer):
                alpha_value: int | float = np.iinfo(rgb_hwc.dtype).max
            elif np.issubdtype(rgb_hwc.dtype, np.floating):
                alpha_value = 1.0
            else:
                raise ValueError(f"Unsupported decoded frame dtype: {rgb_hwc.dtype}")
            alpha = np.full((*rgb_hwc.shape[:2], 1), alpha_value, dtype=rgb_hwc.dtype)
            return np.concatenate([rgb_hwc, alpha], axis=2)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load frame at {pts_seconds:.3f}s from {path_or_url}: {e}") from e
