"""CPU codec adapter with NumPy output and no Torch dependency."""

import importlib.util
from typing import Any, ClassVar

import numpy as np
from tensorcodec.decoders import VideoDecoder

from ..resource_cache import ResourceCache
from .codec_decoder import CodecVideoDecoder, _DecoderState

# tensorcodec itself installs everywhere (image codecs); its video decoder needs the
# platform-specific tensorcodec-av extension, so fail at import rather than on first decode.
if importlib.util.find_spec("tensorcodec_av") is None:
    raise ImportError("tensorcodec-av is not installed")


class TensorCodecVideoDecoder(CodecVideoDecoder):
    """TensorCodec implementation of the cached playback interface."""

    cache: ClassVar[ResourceCache[_DecoderState]] = ResourceCache(max_size=10)
    native_output: ClassVar[bool] = True  # gray, gray12le, gray16le/be, rgb24, rgba
    _open_decoder = staticmethod(VideoDecoder)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        return np.asarray(value)
