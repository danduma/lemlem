"""
Gemini image generation utilities for lemlem.

Provides a thin wrapper around the google-genai client for producing images
from text prompts. Designed to be reusable by editor agents and wiki tooling.
"""
from __future__ import annotations

import logging
from io import BytesIO
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

try:
    from google import genai as _genai
    from google.genai import types as _types
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "google-genai is required for image generation. Install with uv add google-genai."
    ) from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "Pillow is required for image generation. Install with uv add pillow."
    ) from exc

logger = logging.getLogger(__name__)

# Default model must be provided via environment variable or argument.
DEFAULT_GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL") or ""
DEFAULT_IMAGE_FORMAT = os.getenv("GEMINI_IMAGE_FORMAT") or "png"
# Image tuning defaults (configurable via env)
DEFAULT_IMAGE_ASPECT_RATIO = (os.getenv("GEMINI_IMAGE_ASPECT_RATIO") or "16:9").strip()
DEFAULT_IMAGE_SIZE = (os.getenv("GEMINI_IMAGE_SIZE") or "1K").strip()


@dataclass
class ImageGenerationResult:
    """Structured result from an image generation request."""

    path: Path
    width: int
    height: int
    model_used: str
    usage: Optional[dict] = None
    elapsed_ms: Optional[float] = None
    text_annotations: Sequence[str] | None = None
    thoughts: Sequence[str] | None = None


class GeminiImageGenerator:
    """
    Minimal Gemini image generator wrapper.

    Supports text-to-image today and is structured to allow future
    prompt-based refinements of an existing image. Callers should always
    supply an explicit filename under their /images directory so downstream
    agents can persist metadata alongside the generated file.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        response_format: str = DEFAULT_IMAGE_FORMAT,
    ) -> None:
        api_key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("Gemini API key is required (set GEMINI_API_KEY).")

        self.model = (model or DEFAULT_GEMINI_IMAGE_MODEL).strip()
        if not self.model:
            raise ValueError("Gemini image model must be provided (set GEMINI_IMAGE_MODEL).")

        self.response_format = (response_format or DEFAULT_IMAGE_FORMAT).lstrip(".").lower()
        self.image_aspect_ratio = DEFAULT_IMAGE_ASPECT_RATIO
        self.image_size = DEFAULT_IMAGE_SIZE
        self.client = _genai.Client(api_key=api_key)

    def generate_image(
        self,
        prompt: str,
        *,
        save_path: Optional[Path | str] = None,
        extra_prompt: Optional[str] = None,
        reference_image: Optional[Path | str] = None,
        safety_settings: Optional[Iterable[_types.SafetySetting]] = None,
    ) -> ImageGenerationResult:
        """
        Generate an image from a text prompt and save it locally.

        Args:
            prompt: Primary description for the image.
            save_path: Explicit output path. Callers should pass a filename within their images/ directory.
            extra_prompt: Optional secondary text to guide the generation (reserved for future refinements).
            reference_image: Optional existing image to inform the generation (future refinement hook).
            safety_settings: Optional Gemini safety settings sequence.

        Returns:
            ImageGenerationResult with saved path and metadata.
        """
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            raise ValueError("Prompt is required to generate an image.")

        contents = [clean_prompt]
        if extra_prompt:
            contents.append(extra_prompt.strip())

        # Future refinement hook: allow attaching a reference image when provided.
        image_part = self._load_reference_image(reference_image) if reference_image else None
        if image_part:
            contents.append(image_part)

        start = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=_types.GenerateContentConfig(
                thinking_config=_types.ThinkingConfig(include_thoughts=True),
                image_config=_types.ImageConfig(
                    aspect_ratio=self.image_aspect_ratio,
                    image_size=self.image_size,
                ),
                safety_settings=list(safety_settings) if safety_settings else None,
            ),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        image, annotations, thoughts = self._extract_image(response)
        target_path = self._resolve_save_path(save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Let Pillow infer format from extension to avoid incompatibilities.
        image.save(target_path)

        width, height = image.size
        usage = self._safe_usage(response)

        logger.info(
            "Gemini image generated | model=%s | path=%s | size=%sx%s | elapsed_ms=%.2f",
            self.model,
            target_path,
            width,
            height,
            elapsed_ms,
        )

        return ImageGenerationResult(
            path=target_path,
            width=width,
            height=height,
            model_used=self.model,
            usage=usage,
            elapsed_ms=elapsed_ms,
            text_annotations=annotations or None,
            thoughts=thoughts or None,
        )

    def _resolve_save_path(self, save_path: Optional[Path | str]) -> Path:
        if save_path:
            return Path(save_path).expanduser().resolve()

        timestamp = int(time.time() * 1000)
        filename = f"generated_image_{timestamp}.{self.response_format}"
        return Path.cwd() / filename

    def _extract_image(self, response: any) -> tuple[Image.Image, list[str], list[str]]:
        annotations: list[str] = []
        thoughts: list[str] = []
        image: Image.Image | None = None

        # The Gemini SDK exposes `parts` directly on the response for image models.
        for part in getattr(response, "parts", []) or []:
            if getattr(part, "thought", None):
                if getattr(part, "text", None):
                    thoughts.append(str(part.text))
                else:
                    thoughts.append(str(part.thought))
            if getattr(part, "text", None) and not getattr(part, "thought", None):
                annotations.append(str(part.text))
            if getattr(part, "inline_data", None):
                image = self._to_pil_image(part) or image

        # Fall back to candidates if needed
        if image is None:
            for candidate in getattr(response, "candidates", []) or []:
                for part in getattr(candidate, "content", {}).parts or []:
                    if getattr(part, "thought", None):
                        if getattr(part, "text", None):
                            thoughts.append(str(part.text))
                        else:
                            thoughts.append(str(part.thought))
                    if getattr(part, "text", None) and not getattr(part, "thought", None):
                        annotations.append(str(part.text))
                    if getattr(part, "inline_data", None):
                        image = self._to_pil_image(part) or image

        if image is None:
            raise RuntimeError("Gemini response did not contain an image payload.")

        return image, annotations, thoughts

    def _to_pil_image(self, part: any):
        """
        Convert a Gemini inline_data part to a PIL Image.

        Handles both the google-genai helper (as_image) and raw inline bytes.
        """
        # Try native helper first.
        helper = getattr(part, "as_image", None)
        if callable(helper):
            try:
                img = helper()
                if hasattr(img, "size"):
                    return img
            except Exception:
                pass

        inline = getattr(part, "inline_data", None)
        if inline is None:
            return None

        data = getattr(inline, "data", None)
        if not data:
            return None

        # inline_data.data may already be bytes or a base64 string
        try:
            if isinstance(data, str):
                import base64
                raw = base64.b64decode(data)
            else:
                raw = bytes(data)
            return Image.open(BytesIO(raw))
        except Exception:
            return None

    def _safe_usage(self, response: any) -> Optional[dict]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return None

        result: dict[str, int] = {}
        for field in ("prompt_token_count", "candidates_token_count", "total_token_count"):
            value = getattr(usage, field, None)
            if value is not None:
                result[field] = int(value)
        return result or None

    def _load_reference_image(self, reference_image: Path | str) -> Optional[_types.Part]:
        """
        Prepare a reference image for prompt-guided generation.

        Currently a pass-through that enables forward compatibility with
        image-editing or style-transfer flows.
        """
        path = Path(reference_image).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {path}")

        mime_type = self._guess_mime_type(path)
        data = path.read_bytes()
        try:
            return _types.Part.from_bytes(data=data, mime_type=mime_type)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to attach reference image %s: %s", path, exc)
            return None

    @staticmethod
    def _guess_mime_type(path: Path) -> str:
        suffix = path.suffix.lower().lstrip(".")
        if suffix in {"jpg", "jpeg"}:
            return "image/jpeg"
        if suffix == "webp":
            return "image/webp"
        if suffix == "gif":
            return "image/gif"
        return "image/png"


__all__ = [
    "GeminiImageGenerator",
    "ImageGenerationResult",
]


