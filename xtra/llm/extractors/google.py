"""Google Gemini LLM extractor using instructor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from xtra.extractors._image_loader import ImageLoader
from xtra.llm.models import LLMExtractionResult, LLMProvider
from xtra.llm.extractors.openai import _build_prompt

T = TypeVar("T", bound=BaseModel)


def extract_google(
    path: Path | str,
    model: str,
    *,
    schema: Optional[Type[T]] = None,
    prompt: Optional[str] = None,
    pages: Optional[List[int]] = None,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> LLMExtractionResult[Union[T, Dict[str, Any]]]:
    """Extract structured data using Google Gemini."""
    try:
        import instructor
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "Google dependencies not installed. Install with: pip install xtra[llm-google]"
        ) from e

    path = Path(path) if isinstance(path, str) else path
    loader = ImageLoader(path, dpi=dpi)

    try:
        # Configure API key
        genai.configure(api_key=api_key)

        # Load images (Gemini accepts PIL images directly)
        page_nums = pages if pages is not None else list(range(loader.page_count))
        images = [loader.get_page(p) for p in page_nums]

        # Build prompt
        extraction_prompt = _build_prompt(schema, prompt)

        # Create instructor client
        client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=model),
            mode=instructor.Mode.GEMINI_JSON,
        )

        # Build content with images and text
        content = list(images) + [extraction_prompt]

        # Extract with schema or dict
        if schema is not None:
            response = client.generate_content(  # type: ignore
                contents=content,
                response_model=schema,
                max_retries=max_retries,
                generation_config=genai.GenerationConfig(temperature=temperature),
            )
            data = response
        else:
            # For dict extraction, use raw client with JSON instruction
            raw_model = genai.GenerativeModel(model_name=model)
            response = raw_model.generate_content(
                contents=content,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            import json

            data = json.loads(response.text)

        return LLMExtractionResult(
            data=data,
            model=model,
            provider=LLMProvider.GOOGLE,
        )
    finally:
        loader.close()


async def extract_google_async(
    path: Path | str,
    model: str,
    *,
    schema: Optional[Type[T]] = None,
    prompt: Optional[str] = None,
    pages: Optional[List[int]] = None,
    dpi: int = 200,
    max_retries: int = 3,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> LLMExtractionResult[Union[T, Dict[str, Any]]]:
    """Async extract structured data using Google Gemini."""
    try:
        import instructor
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "Google dependencies not installed. Install with: pip install xtra[llm-google]"
        ) from e

    path = Path(path) if isinstance(path, str) else path
    loader = ImageLoader(path, dpi=dpi)

    try:
        # Configure API key
        genai.configure(api_key=api_key)

        # Load images
        page_nums = pages if pages is not None else list(range(loader.page_count))
        images = [loader.get_page(p) for p in page_nums]

        # Build prompt
        extraction_prompt = _build_prompt(schema, prompt)

        # Create instructor client
        client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=model),
            mode=instructor.Mode.GEMINI_JSON,
            use_async=True,
        )

        # Build content with images and text
        content = list(images) + [extraction_prompt]

        # Extract with schema or dict
        if schema is not None:
            response = await client.generate_content(  # type: ignore
                contents=content,
                response_model=schema,
                max_retries=max_retries,
                generation_config=genai.GenerationConfig(temperature=temperature),
            )
            data = response
        else:
            raw_model = genai.GenerativeModel(model_name=model)
            response = await raw_model.generate_content_async(
                contents=content,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            import json

            data = json.loads(response.text)

        return LLMExtractionResult(
            data=data,
            model=model,
            provider=LLMProvider.GOOGLE,
        )
    finally:
        loader.close()
