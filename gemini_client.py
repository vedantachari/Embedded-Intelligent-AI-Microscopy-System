import os
import json
import os
import json
import os
import json
from typing import Optional, Dict, Any

from google import genai
from google.genai import types


class GeminiClient:

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-001"):
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No Gemini API key found. Set GOOGLE_API_KEY environment variable.")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
        return None

    def generate_info(self, class_name: str, extra_context: Optional[str] = None,
                      max_output_tokens: int = 600, temperature: float = 0.0) -> Dict[str, Any]:

        prompt = f"""
You are an expert biologist and microscopy analyst. Produce a DETAILED informational response about the organism/class named "{class_name}" targeted to help a researcher identify it in microscopy/video frames.

Return the result as a single VALID JSON object ONLY (no commentary, no explanation text outside the JSON). The JSON must contain these fields:

- "title": short title (string)
- "summary": 2-4 sentence overview (string)
- "identification": array of 3-8 short bullet points (strings) listing identifying features
- "size": short note about typical size / scale (string) if known, else empty string
- "microscopy_tips": array of 2-6 practical tips to recognize it in microscopy/video (strings)
- "imaging_signatures": array of 1-6 short bullet points about how it appears in images/frames (shape, contrast, motion)

If you need dataset-specific context, use the provided extra context (if any).

Return only valid JSON. Example:
{{"title":"...", "summary":"...", "identification":["...","..."], "size":"...", "microscopy_tips":["..."], "imaging_signatures":["..."], "references":["..."]}}
"""

        if extra_context:
            prompt += "\n\nExtra context: " + extra_context

        config = types.GenerateContentConfig(max_output_tokens=max_output_tokens,
                                             temperature=temperature)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            text = response.text.strip()
        except Exception as e:
            return {"raw": f"[Error calling Gemini API: {e}]"}

        parsed = self._extract_json(text)
        if parsed is not None:
            keys = ["title", "summary", "identification", "size", "microscopy_tips", "imaging_signatures", "references"]
            for k in keys:
                if k not in parsed:
                    parsed[k] = [] if k in ("identification", "microscopy_tips", "imaging_signatures", "references") else ""
            if not isinstance(parsed.get("identification"), list):
                parsed["identification"] = [str(parsed["identification"])] if parsed.get("identification") else []
            if not isinstance(parsed.get("microscopy_tips"), list):
                parsed["microscopy_tips"] = [str(parsed["microscopy_tips"])] if parsed.get("microscopy_tips") else []
            if not isinstance(parsed.get("imaging_signatures"), list):
                parsed["imaging_signatures"] = [str(parsed["imaging_signatures"])] if parsed.get("imaging_signatures") else []
            if not isinstance(parsed.get("references"), list):
                parsed["references"] = [str(parsed["references"])] if parsed.get("references") else []
            return parsed

        return {"raw": text}
