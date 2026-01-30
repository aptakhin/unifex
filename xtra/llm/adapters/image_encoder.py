"""Image encoding for LLM vision APIs."""

from __future__ import annotations

import base64
import io
from typing import List, Literal, Optional

from PIL import Image


class ImageEncoder:
    """Encodes PIL Images to base64 data URLs for LLM vision APIs."""

    def __init__(
        self,
        format: Literal["PNG", "JPEG"] = "PNG",
        quality: int = 85,
        max_dimension: Optional[int] = 2048,
    ) -> None:
        """Initialize encoder.

        Args:
            format: Output image format (PNG or JPEG).
            quality: JPEG quality (1-100), ignored for PNG.
            max_dimension: Max width/height. Images larger are scaled down
                preserving aspect ratio. None to disable resizing.
        """
        self.format = format
        self.quality = quality
        self.max_dimension = max_dimension

    def encode_image(self, image: Image.Image) -> str:
        """Encode a single image to base64 data URL."""
        image = self._resize_if_needed(image)

        buffer = io.BytesIO()
        if self.format == "JPEG":
            # Convert to RGB if necessary (JPEG doesn't support alpha)
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")
            image.save(buffer, format="JPEG", quality=self.quality)
            mime_type = "image/jpeg"
        else:
            image.save(buffer, format="PNG")
            mime_type = "image/png"

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def encode_images(self, images: List[Image.Image]) -> List[str]:
        """Encode multiple images to base64 data URLs."""
        return [self.encode_image(img) for img in images]

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image if it exceeds max dimensions."""
        if self.max_dimension is None:
            return image

        width, height = image.size
        if width <= self.max_dimension and height <= self.max_dimension:
            return image

        scale = min(self.max_dimension / width, self.max_dimension / height)
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS)
