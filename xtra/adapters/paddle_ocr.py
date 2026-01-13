"""Adapter for converting PaddleOCR results to internal schema."""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, field_validator

from xtra.models import TextBlock
from xtra.utils.geometry import polygon_to_bbox_and_rotation

POLYGON_POINTS = 4
COORDINATES_PER_POINT = 2


class PaddleOCRDetection(BaseModel):
    """A single text detection from PaddleOCR.

    PaddleOCR returns detections as [bbox, (text, confidence)] where
    bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] representing 4 corners.
    """

    polygon: List[List[float]]
    text: str
    confidence: float

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != POLYGON_POINTS:
            raise ValueError(f"Polygon must have {POLYGON_POINTS} points, got {len(v)}")
        for point in v:
            if len(point) != COORDINATES_PER_POINT:
                raise ValueError(
                    f"Each point must have {COORDINATES_PER_POINT} coordinates, got {len(point)}"
                )
        return v

    @classmethod
    def from_paddle_format(
        cls, item: Tuple[List[List[float]], Tuple[str, float]]
    ) -> "PaddleOCRDetection":
        """Create from PaddleOCR's native format: [bbox, (text, confidence)]."""
        bbox, (text, confidence) = item
        return cls(polygon=bbox, text=text, confidence=confidence)


class PaddleOCRResult(BaseModel):
    """Validated PaddleOCR result for a single image.

    PaddleOCR returns results as [[[bbox, (text, conf)], ...]] where the outer
    list is for batch processing (always length 1 for single image).
    """

    detections: List[PaddleOCRDetection]

    @classmethod
    def from_paddle_output(cls, result: Optional[list]) -> "PaddleOCRResult":
        """Parse and validate PaddleOCR's raw output format.

        Handles edge cases:
        - None result
        - Empty result [[]]
        - Result with None items [[None]]
        """
        detections: List[PaddleOCRDetection] = []

        if not result or not result[0]:
            return cls(detections=detections)

        for item in result[0]:
            if item is None:
                continue
            detections.append(PaddleOCRDetection.from_paddle_format(item))

        return cls(detections=detections)


class PaddleOCRAdapter:
    """Converts PaddleOCR output to internal schema."""

    def convert_result(self, result: Optional[list]) -> List[TextBlock]:
        """Convert PaddleOCR output to TextBlocks.

        Args:
            result: Raw PaddleOCR output from ocr() method.

        Returns:
            List of TextBlocks with coordinates in pixels.
        """
        validated = PaddleOCRResult.from_paddle_output(result)
        return self._detections_to_blocks(validated.detections)

    def _detections_to_blocks(self, detections: List[PaddleOCRDetection]) -> List[TextBlock]:
        """Convert validated detections to TextBlocks."""
        blocks: List[TextBlock] = []

        for detection in detections:
            if not detection.text or not detection.text.strip():
                continue

            bbox, rotation = polygon_to_bbox_and_rotation(detection.polygon)

            blocks.append(
                TextBlock(
                    text=detection.text,
                    bbox=bbox,
                    rotation=rotation,
                    confidence=detection.confidence,
                )
            )

        return blocks
