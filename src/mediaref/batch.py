"""Batch loading utilities for MediaRef."""

import sys
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Type

import numpy as np
import numpy.typing as npt
from packaging.version import InvalidVersion, Version

from ._internal import NANOSECOND, resolve_video_source

if TYPE_CHECKING:
    from .core import MediaRef
    from .video_decoder import BaseVideoDecoder

# Type alias for decoder backend selection
DecoderBackend = Literal["tensorcodec", "pyav", "torchcodec"]


def _split_by_gap(
    indices: list[int],
    pts_seconds: list[float],
    gap_threshold: float,
) -> list[tuple[list[int], list[float]]]:
    """Split timestamps into contiguous chunks at gaps exceeding *gap_threshold*.

    Returns a list of ``(indices, pts_seconds)`` tuples — one per chunk — sorted by time within each chunk.
    """
    if len(pts_seconds) <= 1:
        return [(indices, pts_seconds)]

    paired = sorted(zip(indices, pts_seconds), key=lambda x: x[1])

    chunks: list[tuple[list[int], list[float]]] = [([paired[0][0]], [paired[0][1]])]
    for i in range(1, len(paired)):
        if paired[i][1] - paired[i - 1][1] > gap_threshold:
            chunks.append(([], []))
        chunks[-1][0].append(paired[i][0])
        chunks[-1][1].append(paired[i][1])
    return chunks


def _coalesce_native_sparse_chunks(
    chunks: list[tuple[list[int], list[float]]],
    backend: DecoderBackend,
    torchcodec_version: Optional[str] = None,
) -> list[tuple[list[int], list[float]]]:
    """Use one native sparse request when the backend optimizes sparse seeks."""
    if len(chunks) <= 1 or backend not in {"tensorcodec", "torchcodec"}:
        return chunks
    if backend == "tensorcodec":
        return [([i for indices, _ in chunks for i in indices], [t for _, times in chunks for t in times])]
    if torchcodec_version is None:
        try:
            torchcodec_version = version("torchcodec")
        except PackageNotFoundError:
            return chunks
    try:
        supports_native_sparse_requests = Version(torchcodec_version) >= Version("0.15")
    except InvalidVersion:
        return chunks
    if not supports_native_sparse_requests:
        return chunks
    return [
        (
            [index for indices, _ in chunks for index in indices],
            [pts for _, timestamps in chunks for pts in timestamps],
        )
    ]


def _decode_video_chunks(
    decoder_class: Type["BaseVideoDecoder"],
    uri: str,
    chunks: list[list[float]],
    decoder_options: Mapping[str, Any],
    storage_options: Optional[Mapping[str, Any]],
) -> list[list[npt.NDArray[np.generic]]]:
    """Decode multiple timestamp chunks from one source with a single decoder open.

    Each chunk is decoded via a separate ``get_frames_played_at`` call so the decoder can seek past
    large gaps instead of decoding every intermediate frame.  Returns one list of RGB HWC frames per
    chunk, in input order.
    """
    source = resolve_video_source(uri)
    options = dict(decoder_options)
    if storage_options is not None:
        options["storage_options"] = storage_options
    with decoder_class(source, **options) as video_decoder:
        native = options.get("output_format") == "native"
        return [
            [
                (f[0] if f.shape[0] == 1 else np.transpose(f, (1, 2, 0))).copy()
                if native
                else np.transpose(f, (1, 2, 0))
                for f in video_decoder.get_frames_played_at(chunk).data
            ]
            for chunk in chunks
        ]


def _get_decoder_class(backend: DecoderBackend) -> Type["BaseVideoDecoder"]:
    """Get decoder class for the specified backend."""
    if backend == "tensorcodec":
        from .video_decoder import TensorCodecVideoDecoder

        return TensorCodecVideoDecoder
    elif backend == "pyav":
        from .video_decoder import PyAVVideoDecoder

        return PyAVVideoDecoder
    elif backend == "torchcodec":
        try:
            from .video_decoder import TorchCodecVideoDecoder

            return TorchCodecVideoDecoder
        except ImportError as e:
            raise ImportError(
                "TorchCodec decoder requested but torchcodec is not installed. "
                "Install with: pip install 'mediaref[torchcodec]'"
            ) from e
    else:
        raise ValueError(f"Unknown decoder backend: {backend}. Must be 'tensorcodec', 'pyav' or 'torchcodec'")


def batch_decode(
    refs: list["MediaRef"],
    decoder: DecoderBackend = "tensorcodec",
    *,
    output_format: Literal["rgb", "native"] = "rgb",
    decoder_options: Optional[Mapping[str, Any]] = None,
    image_decoder: Literal["tensorcodec", "torchcodec"] = "tensorcodec",
    image_decoder_options: Optional[Mapping[str, Any]] = None,
    storage_options: Optional[Mapping[str, Any]] = None,
    allow_images: bool = False,
    allow_multi_video: bool = False,
    gap_threshold: float = 2.0,
    allow_gap: bool = True,
    **kwargs,
) -> list[npt.NDArray[np.generic]]:
    """Decode multiple media references efficiently using batch decoding.

    Groups video frames by file and decodes them in one pass for efficiency.  When timestamps within
    a single video have large gaps, PyAV splits them into contiguous chunks. TensorCodec and recent
    TorchCodec use an optimized native sparse-seek request.

    Args:
        refs: List of MediaRef objects to decode.
        decoder: Decoder backend: ``'tensorcodec'`` (default), ``'pyav'`` or ``'torchcodec'``.
        output_format: ``rgb`` (default) or value-preserving ``native`` for PyAV or TensorCodec
            video refs only. Native grayscale is HW; packed color is HWC.
        decoder_options: Options passed to the selected video decoder constructor.
            For TensorCodec/TorchCodec this includes options such as ``device``, ``seek_mode``,
            ``num_ffmpeg_threads``, and ``dimension_order``. These options apply to
            video refs only.
        image_decoder: Image decoder backend (``'tensorcodec'`` (default) or ``'torchcodec'``).
        image_decoder_options: Options passed to the image backend's ``decode_image``.
        storage_options: Credentials and backend options passed to fsspec for
            image and video URIs.
        allow_images: If ``True``, image refs are accepted and decoded individually.
            If ``False`` (default), image refs raise ``ValueError``.
        allow_multi_video: If ``True``, refs may span multiple video files.
            If ``False`` (default), more than one video URI raises ``ValueError``.
        gap_threshold: Minimum gap in seconds between consecutive sorted timestamps that triggers
            chunking (or an error when *allow_gap* is ``False``).
        allow_gap: If ``True`` (default), gaps exceeding *gap_threshold* cause automatic chunk
            splitting for efficient decoding.  If ``False``, such gaps raise ``ValueError``.
        **kwargs: Additional options forwarded to ``to_ndarray()`` for image loading
            (only used when *allow_images* is ``True``).

    Returns:
        List of RGB numpy arrays in the same order as *refs*.

    Raises:
        ValueError: When a constraint (*allow_images*, *allow_multi_video*, or *allow_gap*)
            is violated.

    Examples:
        >>> refs = [MediaRef(uri="video.mp4", pts_ns=i * 1_000_000_000) for i in range(3)]
        >>> frames = batch_decode(refs)

        >>> # Strict mode: single video, no large gaps
        >>> frames = batch_decode(refs, allow_gap=False)
    """
    if output_format not in {"rgb", "native"}:
        raise ValueError("output_format must be 'rgb' or 'native'")
    options = dict(decoder_options or {})
    if options.get("output_format", output_format) != output_format:
        raise ValueError("Conflicting output_format and decoder_options")
    if output_format == "native":
        if any(not ref.is_video for ref in refs if ref is not None):
            raise ValueError("Native output currently supports video refs only")
        if decoder not in {"pyav", "tensorcodec"}:
            raise ValueError("Native video output requires the PyAV or TensorCodec backend")
        options["output_format"] = "native"
    if not refs:
        return []

    if not isinstance(refs, list):
        raise TypeError(f"refs must be a list, got {type(refs).__name__}")

    if any(ref is None for ref in refs):
        raise ValueError("refs list contains None values")

    video_groups: dict[str, list[tuple[int, "MediaRef"]]] = defaultdict(list)
    image_refs: list[tuple[int, "MediaRef"]] = []

    for i, ref in enumerate(refs):
        if ref.is_video:
            video_groups[ref.uri].append((i, ref))
        else:
            image_refs.append((i, ref))

    if image_refs and not allow_images:
        raise ValueError(
            f"batch_decode() received {len(image_refs)} image reference(s) "
            f"but allow_images=False. Pass allow_images=True to decode images, "
            f"or use ref.to_ndarray() directly."
        )

    if len(video_groups) > 1 and not allow_multi_video:
        uris = list(video_groups.keys())
        raise ValueError(
            f"batch_decode() received refs from {len(uris)} different video files "
            f"but allow_multi_video=False. URIs: {uris}"
        )

    results: list[npt.NDArray[np.generic] | None] = [None] * len(refs)

    # Decode images individually
    for i, ref in image_refs:
        results[i] = ref.to_ndarray(
            image_decoder=image_decoder,
            image_decoder_options=image_decoder_options,
            storage_options=storage_options,
            **kwargs,
        )

    # Decode video frames with gap-aware chunking
    decoder_class = _get_decoder_class(decoder) if video_groups else None
    for uri, group in video_groups.items():
        assert decoder_class is not None
        indices = [i for i, _ in group]

        pts_seconds: list[float] = []
        for _, ref in group:
            if ref.pts_ns is None:
                raise ValueError(f"Video reference missing pts_ns: {ref.uri}")
            pts_seconds.append(ref.pts_ns / NANOSECOND)

        chunks = _split_by_gap(indices, pts_seconds, gap_threshold)

        if len(chunks) > 1 and not allow_gap:
            max_gap = max(chunks[j + 1][1][0] - chunks[j][1][-1] for j in range(len(chunks) - 1))
            raise ValueError(
                f"Timestamps for '{uri}' have a gap of {max_gap:.1f}s "
                f"(threshold: {gap_threshold}s) but allow_gap=False."
            )

        decode_chunks = _coalesce_native_sparse_chunks(chunks, decoder)

        try:
            chunk_pts = [pts for _, pts in decode_chunks]
            decoded = _decode_video_chunks(
                decoder_class,
                uri,
                chunk_pts,
                options,
                storage_options,
            )
            for (chunk_indices, _), frames in zip(decode_chunks, decoded):
                for idx, frame in zip(chunk_indices, frames):
                    results[idx] = frame
        except ImportError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to load batch from '{uri}': {e}") from e

    return results  # type: ignore[return-value]


def cleanup_cache():
    """Clear all cached video decoder state from memory.

    This function should be called when you're done with batch decoding
    to free up resources. It's automatically called on process exit.

    Optional backends are not imported just to clean them up. Calling this
    without PyAV or TorchCodec MUST NOT raise, so callers may invoke it
    defensively without knowing which extras are present.

    Examples:
        >>> from mediaref import MediaRef, batch_decode, cleanup_cache
        >>>
        >>> # Decode many frames
        >>> refs = [MediaRef(uri="video.mp4", pts_ns=i * 1_000_000_000) for i in range(100)]
        >>> frames = batch_decode(refs)
        >>>
        >>> # Clean up when done
        >>> cleanup_cache()
    """
    module = sys.modules.get("mediaref.cached_av")
    if module is not None:
        module.cleanup_cache()

    # Do not import an optional backend just to clean it up. If a backend was
    # used, its module is already loaded and its cache can be cleared directly.
    module = sys.modules.get("mediaref.video_decoder.pyav_decoder")
    if module is not None:
        module.clear_metadata_cache()
    module = sys.modules.get("mediaref.video_decoder.torchcodec_decoder")
    if module is not None:
        module.TorchCodecVideoDecoder.clear_cache()

    module = sys.modules.get("mediaref.video_decoder.tensorcodec_decoder")
    if module is not None:
        module.TensorCodecVideoDecoder.clear_cache()
