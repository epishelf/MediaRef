"""PyAV-based video decoder with TorchCodec-compatible playback semantics."""

import copy
import gc
import os
import threading
import warnings
from collections import OrderedDict
from fractions import Fraction
from typing import Any, List, Mapping, Optional

import av
import numpy as np
import numpy.typing as npt

from .. import cached_av
from .._internal import _FILE_URI_PREFIX, _file_uri_to_path, is_cloud_uri, make_cache_key
from .._typing import PathLike
from .base import BaseVideoDecoder
from .frame_batch import FrameBatch
from .types import VideoStreamMetadata

# Garbage collection interval for PyAV reference cycles
# Reference: https://github.com/pytorch/vision/blob/428a54c96e82226c0d2d8522e9cbfdca64283da0/torchvision/io/video.py#L53-L55
_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 10

# Local containers are opened per decoder (cheap), but metadata extraction decodes
# a frame, so cache it keyed on file identity. Remote containers are cached in
# cached_av instead, where reopening costs seconds of network round trips.
_METADATA_CACHE_SIZE = 256
_metadata_cache: "OrderedDict[tuple, VideoStreamMetadata]" = OrderedDict()
_metadata_lock = threading.Lock()


def clear_metadata_cache() -> None:
    """Drop cached local-file metadata."""
    with _metadata_lock:
        _metadata_cache.clear()


# Threshold for sparse query detection (seconds between consecutive timestamps)
_SPARSE_QUERY_GAP_THRESHOLD = 1.0


def _frame_to_rgba(
    frame: av.VideoFrame, reformatter: "av.video.reformatter.VideoReformatter | None" = None
) -> npt.NDArray[np.uint8]:
    """Convert PyAV frame to RGBA numpy array.

    NOTE: Convert ARGB to RGBA manually instead of using `to_ndarray(format="rgba")`.
    Direct RGBA conversion causes memory corruption on certain videos:
      Error: "malloc_consolidate(): invalid chunk size; Fatal Python error: Aborted"
      Example: https://huggingface.co/datasets/open-world-agents/example_dataset/resolve/main/example.mkv (pts_ns=1_000_000_000)
    Possibly related to:
      - PyAV issue: https://github.com/PyAV-Org/PyAV/issues/1269
      - FFmpeg ticket: https://trac.ffmpeg.org/ticket/9254

    Args:
        frame: PyAV VideoFrame to convert
        reformatter: Optional shared ``av.video.reformatter.VideoReformatter``.
            PyAV's ``to_ndarray(format=...)`` sets up a fresh reformatter on every
            call; at small resolutions that per-call setup dominates the pixel work
            (~10x measured at 128x128), so batch callers pass one reformatter to
            amortize it. Output is byte-identical to the stateless path, including
            across varying frame shapes/formats through one reformatter (covered by
            tests; only the reuse benefit degrades when parameters change).

    Returns:
        RGBA numpy array (H, W, 4) with uint8 dtype
    """
    if reformatter is not None:
        argb_array = reformatter.reformat(frame, format="argb").to_ndarray()
    else:
        argb_array = frame.to_ndarray(format="argb")
    # ARGB format stores channels as [A, R, G, B], so we reorder to [R, G, B, A]
    rgba_array: npt.NDArray[np.uint8] = argb_array[:, :, [1, 2, 3, 0]]
    return rgba_array


def _convert_av_frames_to_nchw(av_frames: List[av.VideoFrame]) -> List[npt.NDArray[np.uint8]]:
    """Convert a list of PyAV frames to NCHW numpy arrays (RGB)."""
    # One SwsContext for the whole batch instead of one per frame (see _frame_to_rgba).
    reformatter = av.video.reformatter.VideoReformatter()
    frames = []
    for frame in av_frames:
        rgba_array = _frame_to_rgba(frame, reformatter)
        frame_nchw = np.transpose(rgba_array[:, :, :3], (2, 0, 1)).astype(np.uint8)
        frames.append(frame_nchw)
    return frames


class PyAVVideoDecoder(BaseVideoDecoder):
    """Video decoder using PyAV with TorchCodec-compatible playback semantics.

    Implements the playback model where a frame is displayed from its pts until
    the next frame's pts. For timestamp t, returns frame[i] where:
        frame[i].pts <= t < frame[i+1].pts

    Args:
        source: Local path, fsspec URI, or a binary file-like object.
        storage_options: Credentials and backend options passed to fsspec.
        output_format: ``rgb`` (default) or ``native``. Native output preserves
            decoded sample values in NCHW; unsupported planar/packed formats fail.
        expected_pixel_format: Optional native source-format assertion.

    Examples:
        >>> with PyAVVideoDecoder("video.mp4") as decoder:
        ...     batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
        ...     print(batch.data.shape)  # (3, 3, H, W)
    """

    def __init__(
        self,
        source: PathLike,
        *,
        storage_options: Optional[Mapping[str, Any]] = None,
        output_format: str = "rgb",
        expected_pixel_format: Optional[str] = None,
        **kwargs,
    ):
        """Initialize PyAV video decoder."""
        if isinstance(source, str) and source.startswith(_FILE_URI_PREFIX):
            source = _file_uri_to_path(source)  # FFmpeg cannot open file:///C:/... on Windows
        super().__init__(source, **kwargs)
        if output_format not in {"rgb", "native"}:
            raise ValueError("output_format must be 'rgb' or 'native'")
        if expected_pixel_format is not None and output_format != "native":
            raise ValueError("expected_pixel_format requires native output")
        self.output_format = output_format
        self.expected_pixel_format = expected_pixel_format
        remote = isinstance(source, str) and is_cloud_uri(source)
        self._container = cached_av.open(
            source,
            "r",
            keep_av_open=remote,
            storage_options=storage_options,
            **kwargs,
        )
        self._closed = False
        try:
            self._metadata = self._cached_metadata(source, kwargs) if not remote else self._extract_metadata()
        except Exception:
            self.close()
            raise

    def _cached_metadata(self, source: PathLike, kwargs: Mapping[str, Any]) -> VideoStreamMetadata:
        try:
            stat = os.stat(source)  # type: ignore[arg-type]
        except (OSError, TypeError, ValueError):  # file-likes, file:// URIs
            return self._extract_metadata()
        key = (make_cache_key(os.fspath(source), kwargs), stat.st_mtime_ns, stat.st_size)
        with _metadata_lock:
            metadata = _metadata_cache.get(key)
            if metadata is not None:
                _metadata_cache.move_to_end(key)
                return copy.copy(metadata)
        metadata = self._extract_metadata()
        with _metadata_lock:
            _metadata_cache[key] = metadata
            while len(_metadata_cache) > _METADATA_CACHE_SIZE:
                _metadata_cache.popitem(last=False)
        return copy.copy(metadata)

    def _extract_metadata(self) -> VideoStreamMetadata:
        """Extract video stream metadata from container.

        Stream boundary strategy (cf. TorchCodec's two modes):
          - TorchCodec exact mode: scans ALL packets → begin=min(PTS), end=max(PTS+dur).
            Accurate but O(N) and requires full file read for remote URLs.
          - TorchCodec approximate mode: header-based → begin=start_time, end=start_time+dur.
            O(1) but may be loose (container duration > actual content).

        We use a hybrid: content-based begin (decode first frame) + header-based end
        (stream.start_time + duration). Begin must be content-based for accurate query
        validation. End uses header because reliably finding the last frame via seek is
        fragile (MKV with sparse keyframes returns 0 frames when seeking past the last
        cluster boundary). The loose end bound is safe: queries past the last frame PTS
        but within the header end return the last frame per playback semantics.

        IMPORTANT: end must use stream.start_time (header), NOT first_pts (content).
        Mixing sources inflates end when first_pts > start_time (e.g. first_pts=0.033
        but start_time=0 → end was 1 frame too large).
        """
        container = self._container
        if not container.streams.video:
            raise ValueError(f"No video streams found in {self.source}")
        stream = container.streams.video[0]

        if stream.average_rate:
            average_rate = Fraction(stream.average_rate)
        else:
            raise ValueError("Failed to determine average rate")

        if stream.duration and stream.time_base:
            duration_seconds = Fraction(stream.duration * stream.time_base)
        elif container.duration:
            duration_seconds = Fraction(container.duration, av.time_base)
        else:
            raise ValueError("Failed to determine duration")

        # begin: content-based (first decoded frame PTS)
        container.seek(0)
        first_pts: Fraction | None = None
        pixel_format = None
        for frame in container.decode(video=0):
            pixel_format = frame.format.name
            if frame.time is not None:
                first_pts = Fraction(frame.time).limit_denominator(1000000)
            break

        if first_pts is None:
            raise ValueError("No video frames found")

        container.seek(0)

        begin_stream_seconds = first_pts

        # end: header-based (stream.start_time + duration)
        if stream.start_time is not None and stream.time_base:
            stream_start = Fraction(stream.start_time * stream.time_base)
        else:
            stream_start = Fraction(0)
        end_stream_seconds = stream_start + duration_seconds

        num_frames = stream.frames if stream.frames else int(duration_seconds * average_rate)

        return VideoStreamMetadata(
            num_frames=num_frames,
            duration_seconds=duration_seconds,
            average_rate=average_rate,
            width=stream.width,
            height=stream.height,
            begin_stream_seconds=begin_stream_seconds,
            end_stream_seconds=end_stream_seconds,
            pixel_format=pixel_format,
        )

    @property
    def metadata(self) -> VideoStreamMetadata:
        """Access video stream metadata."""
        return self._metadata

    def _create_empty_batch(self) -> FrameBatch:
        """Create an empty FrameBatch with correct spatial dimensions."""
        channels, dtype = self._output_layout(self._metadata.pixel_format)
        return FrameBatch(
            data=np.empty((0, channels, self._metadata.height, self._metadata.width), dtype=dtype),
            pts_seconds=np.array([], dtype=np.float64),
            duration_seconds=np.array([], dtype=np.float64),
            pixel_format=self._metadata.pixel_format if self.output_format == "native" else "rgb24",
        )

    def _output_layout(self, pixel_format):
        if self.output_format == "rgb":
            return 3, np.uint8
        if self.expected_pixel_format is not None and pixel_format != self.expected_pixel_format:
            raise ValueError(f"Expected source {self.expected_pixel_format}, got {pixel_format}")
        # Only layouts with an unambiguous value-preserving NCHW representation.
        layouts = {
            "gray": (1, np.uint8),
            "gray12le": (1, np.uint16),
            "gray16le": (1, np.uint16),
            "gray16be": (1, np.uint16),
            "rgb24": (3, np.uint8),
            "rgba": (4, np.uint8),
        }
        if pixel_format not in layouts:
            raise ValueError(f"Native output does not support pixel format {pixel_format!r}")
        return layouts[pixel_format]

    def _convert_frames(self, frames):
        if self.output_format == "rgb":
            return _convert_av_frames_to_nchw(frames)
        result = []
        for frame in frames:
            if frame.format.name != self._metadata.pixel_format:
                raise ValueError("Pixel format changed within stream")
            channels, _ = self._output_layout(frame.format.name)
            array = frame.to_ndarray()  # No color conversion, rescaling or quantization.
            result.append(array[None] if channels == 1 else array.transpose(2, 0, 1))
        return result

    def get_frames_played_at(self, seconds: List[float]) -> FrameBatch:
        """Retrieve frames that would be displayed at specific timestamps.

        Follows TorchCodec's playback semantics: returns the frame where
        frame[i].pts <= timestamp < frame[i+1].pts.

        Args:
            seconds: List of timestamps in seconds

        Returns:
            FrameBatch with frame data in NCHW format

        Raises:
            ValueError: If any timestamp is outside [begin_stream_seconds, end_stream_seconds)
        """
        if not seconds:
            return self._create_empty_batch()

        # Validate timestamps per playback_semantics.md boundary conditions
        begin_stream = float(self._metadata.begin_stream_seconds)
        end_stream = float(self._metadata.end_stream_seconds)  # type: ignore[arg-type]
        for t in seconds:
            if t < begin_stream:
                raise ValueError(f"Timestamp {t}s < begin_stream_seconds ({begin_stream}s)")
            if t >= end_stream:
                raise ValueError(f"Timestamp {t}s >= end_stream_seconds ({end_stream}s)")

        # Get frames using playback semantics
        av_frames = self._get_frames_played_at(seconds)

        # Convert to RGB numpy arrays in NCHW format
        frames = self._convert_frames(av_frames)

        pts_list = [float(frame.time) for frame in av_frames]
        duration = float(1.0 / self._metadata.average_rate)

        return FrameBatch(
            data=np.stack(frames, axis=0),
            pts_seconds=np.array(pts_list, dtype=np.float64),
            duration_seconds=np.full(len(seconds), duration, dtype=np.float64),
            pixel_format=self._metadata.pixel_format if self.output_format == "native" else "rgb24",
        )

    def get_frames_played_in_range(
        self, start_seconds: float, stop_seconds: float, fps: Optional[float] = None
    ) -> FrameBatch:
        """Return multiple frames in the given range [start_seconds, stop_seconds).

        Args:
            start_seconds: Time, in seconds, of the start of the range.
            stop_seconds: Time, in seconds, of the end of the range (excluded).
            fps: If specified, resample output to this frame rate by
                duplicating or dropping frames as necessary. If None,
                returns frames at the source video's frame rate.

        Returns:
            FrameBatch with frame data in NCHW format.

        Raises:
            ValueError: If the range parameters are invalid.
        """
        begin_stream = float(self._metadata.begin_stream_seconds)
        end_stream = float(self._metadata.end_stream_seconds)

        if not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be less than or equal to stop seconds ({stop_seconds})."
            )
        if not begin_stream <= start_seconds < end_stream:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be greater than or equal to {begin_stream} "
                f"and less than {end_stream}."
            )
        if not stop_seconds <= end_stream:
            raise ValueError(f"Invalid stop seconds: {stop_seconds}. It must be less than or equal to {end_stream}.")

        if fps is not None:
            # Resample: generate timestamps at the given fps and get frames
            timestamps = np.arange(start_seconds, stop_seconds, 1.0 / fps).tolist()
            if not timestamps:
                return self._create_empty_batch()
            return self.get_frames_played_at(timestamps)

        # Native frame rate: decode all frames with pts in [start_seconds, stop_seconds)
        self._seek_to_or_before(start_seconds)

        av_frames: List[av.VideoFrame] = []
        for frame in self._container.decode(video=0):
            if frame.time is None:
                raise ValueError("Frame time is None")
            frame_pts = float(frame.time)
            if frame_pts >= stop_seconds:
                break
            if frame_pts >= start_seconds:
                av_frames.append(frame)

        if not av_frames:
            return self._create_empty_batch()

        frames = self._convert_frames(av_frames)

        pts_list = [float(frame.time) for frame in av_frames]
        duration = float(1.0 / self._metadata.average_rate)

        return FrameBatch(
            data=np.stack(frames, axis=0),
            pts_seconds=np.array(pts_list, dtype=np.float64),
            duration_seconds=np.full(len(av_frames), duration, dtype=np.float64),
            pixel_format=self._metadata.pixel_format if self.output_format == "native" else "rgb24",
        )

    def _get_frames_played_at(self, seconds: List[float]) -> List[av.VideoFrame]:
        """Get frames using TorchCodec playback semantics.

        For each timestamp, returns the frame where:
            frame[i].pts <= timestamp < frame[i+1].pts

        This is the "frame being played at" semantic - the frame that would
        be displayed on screen at the given timestamp.
        """
        global _CALLED_TIMES
        _CALLED_TIMES += 1
        if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == 0:
            gc.collect()

        # Sort queries for efficient sequential access while preserving output order
        indexed_queries = sorted(enumerate(seconds), key=lambda x: x[1])
        results: List[av.VideoFrame] = [None] * len(seconds)  # type: ignore

        # Warn if consecutive sorted timestamps are far apart (single-pass is wasteful)
        if len(indexed_queries) >= 2:
            sorted_times = [t for _, t in indexed_queries]
            max_gap = max(b - a for a, b in zip(sorted_times, sorted_times[1:]))
            if max_gap > _SPARSE_QUERY_GAP_THRESHOLD:
                warnings.warn(
                    f"Sparse timestamp query detected: max gap between consecutive "
                    f"timestamps is {max_gap:.1f}s (threshold: {_SPARSE_QUERY_GAP_THRESHOLD}s). "
                    f"Single-pass decoding will decode all intermediate frames. "
                    f"For widely spaced timestamps, consider per-timestamp seeking instead.",
                    UserWarning,
                    stacklevel=3,
                )

        query_idx = 0
        prev_frame: av.VideoFrame | None = None
        prev_frame_pts: float = float("-inf")

        # Smart seek: try to seek to first query, fall back to start if we overshoot
        first_query_time = indexed_queries[0][1]
        self._seek_to_or_before(first_query_time)

        # Decode frames and match to queries using nextPts logic
        for frame in self._container.decode(video=0):
            if frame.time is None:
                raise ValueError("Frame time is None")

            frame_pts = float(frame.time)

            # Process all queries where: prev_frame.pts <= query < frame.pts
            while query_idx < len(indexed_queries):
                orig_idx, query_time = indexed_queries[query_idx]
                if query_time < frame_pts:
                    # This query's timestamp is before current frame's pts
                    # Verify prev_frame.pts <= query_time (should always be true if validated)
                    if prev_frame is not None and prev_frame_pts <= query_time:
                        results[orig_idx] = prev_frame
                        query_idx += 1
                    elif prev_frame is None:
                        # Query is before first frame - this should have been caught by validation
                        raise ValueError(f"Timestamp {query_time}s is before first frame (pts={frame_pts}s)")
                    else:
                        # prev_frame_pts > query_time - should not happen with correct validation
                        raise ValueError(
                            f"Internal error: query {query_time}s not in range [{prev_frame_pts}, {frame_pts})"
                        )
                else:
                    break

            prev_frame = frame
            prev_frame_pts = frame_pts

            # Check if we've processed all queries
            if query_idx >= len(indexed_queries):
                break

        # Handle remaining queries (timestamps at or after last decoded frame)
        # These should satisfy: prev_frame.pts <= query < end_stream_seconds
        while query_idx < len(indexed_queries):
            orig_idx, query_time = indexed_queries[query_idx]
            if prev_frame is not None and prev_frame_pts <= query_time:
                results[orig_idx] = prev_frame
                query_idx += 1
            else:
                raise ValueError(f"Could not find frame for timestamp {query_time}s")

        # Verify all queries were satisfied
        for i, result in enumerate(results):
            if result is None:
                raise ValueError(f"Could not find frame for timestamp {seconds[i]}s")

        return results

    def _seek_to_or_before(self, target_seconds: float) -> None:
        """Seek to target time or before it, using exponential backoff if needed.

        PyAV seeks to keyframes, which may land past the target if keyframes are sparse.
        This method detects overshooting and backs off exponentially until we find a
        valid seek point before the target.
        """
        stream = self._container.streams.video[0]
        time_base = float(stream.time_base)
        begin_stream = float(self._metadata.begin_stream_seconds)

        # Start with seeking to the target
        seek_target = target_seconds
        buffer = 1.0  # Initial backoff buffer in seconds

        while True:
            seek_pts = int(seek_target / time_base)
            self._container.seek(seek_pts, stream=stream, any_frame=False, backward=True)

            # Check if we overshot by peeking at the first frame
            # No frame means the seek landed on a packet the decoder cannot start
            # from (e.g. MKV flags every HEVC packet as a keyframe); back off.
            frame = next(self._container.decode(video=0), None)

            if frame is not None and frame.time is not None and frame.time <= target_seconds:
                # Good! We landed at or before the target
                # Re-seek to restore position (we consumed one frame)
                self._container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
                return

            # We overshot! Back off and try again
            seek_target = target_seconds - buffer
            buffer *= 2  # Exponential backoff

            # If we've backed off past the stream start, seek to start
            if seek_target <= begin_stream:
                self._container.seek(0)
                return

    def close(self):
        """Release video decoder resources."""
        if not self._closed and hasattr(self, "_container"):
            self._closed = True
            self._container.close()
