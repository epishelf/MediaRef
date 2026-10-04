"""DataURI class for handling data URI encoding and decoding."""

import base64
import io
import mimetypes
from pathlib import Path
from typing import Literal, Optional, Union
from urllib.parse import quote, unquote, urlparse

import numpy as np
import numpy.typing as npt
import PIL.Image
from pydantic import BaseModel, Field, model_validator
from tensorcodec.encoders import JpegEncoder

from ._internal import decode_image_as_rgba, rgba_to_format

# ============================================================================
# Internal image encoding/decoding
# ============================================================================


def _encode_image_to_bytes(
    array: npt.NDArray[np.uint8],
    format: Literal["png", "jpeg", "bmp"],
    quality: Optional[int] = None,
) -> bytes:
    """Encode an RGBA (H, W, 4) uint8 array; only PNG keeps the alpha channel.

    JPEG uses TensorCodec's encoder. PNG (RGBA) and BMP (RGB) use Pillow: TensorCodec's
    encoders take only 1 or 3 channels and have no BMP writer.
    """
    try:
        if format == "jpeg":
            if quality is None:
                quality = 85
            if not (1 <= quality <= 100):
                raise ValueError("JPEG quality must be between 1 and 100")
            rgb_chw = np.ascontiguousarray(array[..., :3].transpose(2, 0, 1))
            return JpegEncoder(rgb_chw).to_tensor(quality=quality).tobytes()
        if format == "png":
            pil_image, pil_format = PIL.Image.fromarray(np.ascontiguousarray(array)), "PNG"
        elif format == "bmp":
            pil_image, pil_format = PIL.Image.fromarray(np.ascontiguousarray(array[..., :3])), "BMP"
        else:
            raise ValueError(f"Unsupported format: {format}")
        buffer = io.BytesIO()
        pil_image.save(buffer, format=pil_format)
        return buffer.getvalue()
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to encode image as {format}: {e}") from e


def _decode_image_to_rgba(image_bytes: bytes) -> npt.NDArray[np.uint8]:
    """Decode image bytes to RGBA (H, W, 4) with TensorCodec; opaque alpha is added if absent."""
    try:
        return decode_image_as_rgba(image_bytes)  # type: ignore[return-value]
    except Exception as e:
        raise ValueError(f"Failed to decode image data: {e}") from e


class DataURI(BaseModel):
    """Data URI handler for encoding and decoding media.

    Supports RFC 2397 data URI scheme for embedding media data directly in URIs.

    Encoding Requirements:
        - Binary data (images, etc.): Use base64 encoding (automatic in from_image/from_file)
        - Text data: Must be percent-encoded if it contains reserved characters, spaces,
          newlines, or other non-printing characters (RFC 3986)

    Examples:
        >>> # From numpy array
        >>> import numpy as np
        >>> array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> data_uri = DataURI.from_image(array, format="png")
        >>> print(data_uri.uri)
        >>>
        >>> # From file
        >>> data_uri = DataURI.from_file("image.png")
        >>> print(data_uri.mimetype)  # "image/png"
        >>>
        >>> # Parse existing data URI
        >>> uri_str = "data:image/png;base64,iVBORw0KG..."
        >>> data_uri = DataURI.from_uri(uri_str)
        >>> array = data_uri.to_ndarray()  # Default RGB format
    """

    mimetype: str = Field(description="MIME type (e.g., 'image/png')")
    is_base64: bool = Field(default=True, description="Whether data is base64 encoded")
    data: bytes = Field(description="Data payload (base64 string as bytes if is_base64=True, raw bytes otherwise)")

    @model_validator(mode="after")
    def _validate_data_encoding(self) -> "DataURI":
        """Validate that non-base64 data is properly URL-encoded."""
        if self.is_base64:
            return self

        try:
            text_data = self.data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Non-base64 data must be valid UTF-8 text: {e}") from e

        # Check if data is properly URL-encoded by:
        # 1. Unquote it to get the original data
        # 2. Re-quote it and compare with the input
        # This ensures the data is already properly encoded
        try:
            unquoted = unquote(text_data)
            requoted = quote(unquoted, safe="")
            if text_data != requoted:
                raise ValueError(
                    "Non-base64 data URI contains unquoted characters. "
                    "Data should be URL-encoded before creating DataURI, or use base64 encoding."
                )
        except Exception as e:
            raise ValueError(f"Invalid URL encoding in non-base64 data: {e}") from e

        return self

    # ========== Properties ==========

    @property
    def decoded_data(self) -> bytes:
        """Get the decoded data payload.

        If data is base64 encoded, this decodes it. Otherwise returns raw data.

        Returns:
            Decoded bytes
        """
        if self.is_base64:
            return base64.b64decode(self.data)
        else:
            return self.data

    @property
    def uri(self) -> str:
        """Construct and return the full data URI string.

        Returns:
            Data URI string in format: data:[mimetype];base64,[data]
        """
        data_str = self.data.decode("utf-8")
        return f"data:{self.mimetype};base64,{data_str}" if self.is_base64 else f"data:{self.mimetype},{data_str}"

    @property
    def is_image(self) -> bool:
        """True if MIME type is image/*.

        Returns:
            True if mimetype starts with 'image/'
        """
        return self.mimetype.startswith("image/")

    # ========== Class Methods for Construction ==========

    @classmethod
    def from_uri(cls, uri: str) -> "DataURI":
        """Create DataURI from a data URI string.

        Args:
            uri: Data URI string (e.g., "data:image/png;base64,...")

        Returns:
            DataURI instance

        Raises:
            ValueError: If URI is invalid or not a data URI
        """
        parsed = urlparse(uri)
        if parsed.scheme != "data":
            raise ValueError(f"Invalid data URI scheme: {parsed.scheme}")

        try:
            # Split on first comma to separate metadata from data
            metadata, data_part = parsed.path.split(",", 1)

            # Parse metadata
            parts = metadata.split(";")
            mimetype = parts[0] if parts[0] else "text/plain"
            is_base64 = "base64" in parts

            # Store data as-is, expecting input to be correct based on spec
            data_bytes = data_part.encode("utf-8")

            return cls(mimetype=mimetype, is_base64=is_base64, data=data_bytes)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid data URI format: {e}") from e

    @classmethod
    def from_image(
        cls,
        image: Union[npt.NDArray[np.uint8], PIL.Image.Image],
        format: Literal["png", "jpeg", "bmp"] = "png",
        quality: Optional[int] = None,
        input_format: Literal["rgb", "bgr", "rgba", "bgra"] = "rgb",
    ) -> "DataURI":
        """Create from numpy array or PIL Image.

        Args:
            image: PIL Image or numpy array
            format: Output format ('png', 'jpeg', 'bmp')
            quality: JPEG quality (1-100), ignored for PNG/BMP
            input_format: Input channel order for numpy arrays. Default: 'rgb'.
                - 'rgb': RGB format (3 channels)
                - 'bgr': BGR format (3 channels, e.g., from cv2.imread)
                - 'rgba': RGBA format (4 channels)
                - 'bgra': BGRA format (4 channels, e.g., from cv2.imread with alpha)
                Ignored for PIL Images.

        Returns:
            DataURI instance

        Note:
            Alpha channel is only preserved in PNG format.

        Examples:
            >>> # RGB numpy array (default)
            >>> rgb_array = np.zeros((100, 100, 3), dtype=np.uint8)
            >>> data_uri = DataURI.from_image(rgb_array, format="png")
            >>>
            >>> # BGR numpy array (e.g., from your own OpenCV code)
            >>> bgr_array = cv2.imread("image.jpg")
            >>> data_uri = DataURI.from_image(bgr_array, format="png", input_format="bgr")
        """
        if isinstance(image, PIL.Image.Image):
            rgba_array: npt.NDArray[np.uint8] = np.array(image.convert("RGBA"), dtype=np.uint8)
        else:
            if image.ndim != 3:
                raise ValueError(f"Expected 3D array (H, W, C), got shape {image.shape}")

            channels = image.shape[2]
            if channels == 3:
                # Convert to RGBA based on input format
                if input_format in ("rgb", "bgr"):
                    rgb = image if input_format == "rgb" else image[..., ::-1]
                    opaque = np.full((*image.shape[:2], 1), 255, dtype=image.dtype)
                    rgba_array = np.concatenate([rgb, opaque], axis=-1)
                else:
                    raise ValueError(
                        f"Invalid input_format '{input_format}' for 3-channel array. Must be 'rgb' or 'bgr'"
                    )
            elif channels == 4:
                # Convert to RGBA based on input format
                if input_format == "rgb" or input_format == "rgba":
                    rgba_array = image  # Assume RGBA
                elif input_format == "bgr" or input_format == "bgra":
                    rgba_array = image[..., [2, 1, 0, 3]]
                else:
                    raise ValueError(
                        f"Invalid input_format '{input_format}' for 4-channel array. "
                        f"Must be 'rgb', 'bgr', 'rgba', or 'bgra'"
                    )
            else:
                raise ValueError(f"Expected 3 or 4 channels, got {channels}")

        image_bytes = _encode_image_to_bytes(rgba_array, format=format, quality=quality)
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        mimetype = f"image/{format}"
        return cls(mimetype=mimetype, is_base64=True, data=base64_str.encode("utf-8"))

    @classmethod
    def from_file(cls, path: Union[str, Path], format: Optional[str] = None) -> "DataURI":
        """Create from file path (auto-detect format if not specified).

        Args:
            path: File path
            format: Optional format override (e.g., "png", "jpeg")

        Returns:
            DataURI instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format cannot be determined
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Determine MIME type
        if format:
            mimetype = f"image/{format}"
        else:
            # Auto-detect from file extension
            guessed_type = mimetypes.guess_type(str(path_obj))[0]
            if guessed_type:
                mimetype = guessed_type
            else:
                # Default to application/octet-stream if unknown
                mimetype = "application/octet-stream"

        # Read file data
        with open(path_obj, "rb") as f:
            raw_data = f.read()

        # Store as base64 encoded string (as bytes)
        base64_str = base64.b64encode(raw_data).decode("utf-8")
        return cls(mimetype=mimetype, is_base64=True, data=base64_str.encode("utf-8"))

    # ========== Conversion Methods ==========

    def to_ndarray(
        self,
        format: Literal["rgb", "bgr", "rgba", "bgra", "gray"] = "rgb",
    ) -> npt.NDArray[np.uint8]:
        """Convert to numpy ndarray in specified format.

        Args:
            format: Output format (default: "rgb")
                - "rgb": RGB color (H, W, 3)
                - "bgr": BGR color (H, W, 3)
                - "rgba": RGB with alpha (H, W, 4)
                - "bgra": BGR with alpha (H, W, 4)
                - "gray": Grayscale (H, W)

        Returns:
            Numpy ndarray in requested format

        Raises:
            ValueError: If data is not a valid image or format is invalid

        Examples:
            >>> data_uri = DataURI.from_file("image.png")
            >>> rgb = data_uri.to_ndarray()  # Default RGB format
        """
        if not self.is_image:
            raise ValueError(f"Cannot convert non-image MIME type '{self.mimetype}' to numpy array")

        try:
            # Get decoded data (handles base64 decoding if needed)
            image_bytes = self.decoded_data

            rgba_array = _decode_image_to_rgba(image_bytes)

            return rgba_to_format(rgba_array, format)
        except ValueError:
            # Re-raise ValueError (format errors or conversion errors)
            raise
        except Exception as e:
            raise ValueError(f"Failed to decode image data: {e}") from e

    def to_pil_image(self) -> PIL.Image.Image:
        """Convert to PIL Image.

        Returns:
            PIL Image object

        Raises:
            ValueError: If data is not a valid image
        """
        # Convert to RGB array first, then to PIL
        rgb_array = self.to_ndarray(format="rgb")
        return PIL.Image.fromarray(rgb_array, mode="RGB")

    # ========== String Representation ==========

    def __str__(self) -> str:
        """Return the full data URI string (same as .uri property).

        Returns:
            Data URI string
        """
        return self.uri

    def __len__(self) -> int:
        """Return the size of the decoded data in bytes (same as .size property).

        Returns:
            Size of the decoded data in bytes
        """
        return len(self.uri)
