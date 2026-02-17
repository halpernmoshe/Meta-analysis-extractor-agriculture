"""
LLM Client wrapper for Anthropic Claude and Google Gemini APIs

Handles API calls with retry logic, error handling, and response parsing.
Supports both providers with a unified interface.
"""
import json
import re
import time
from typing import Optional, Dict, Any, List

# Import Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Import Google GenAI (new SDK)
try:
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from config import (
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    RECON_MODEL,
    EXTRACT_MODEL,
    GEMINI_RECON_MODEL,
    GEMINI_EXTRACT_MODEL,
    RECON_TEMPERATURE,
    EXTRACT_TEMPERATURE,
    MAX_TOKENS_RECON,
    MAX_TOKENS_EXTRACT,
    MAX_RETRIES,
    RETRY_DELAY
)


class LLMClient:
    """
    Unified wrapper for Anthropic Claude and Google Gemini APIs

    Provides methods for different types of calls:
    - call_recon: For reconnaissance tasks (can use cheaper model)
    - call_extract: For extraction tasks (needs accuracy)

    Supports both providers with automatic switching based on config.
    """

    def __init__(self, api_key: str = None, provider: str = None, google_api_key: str = None):
        self.provider = provider or LLM_PROVIDER

        # Initialize based on provider
        if self.provider == "google":
            self._init_google(google_api_key or api_key or GOOGLE_API_KEY)
            self.recon_model = GEMINI_RECON_MODEL
            self.extract_model = GEMINI_EXTRACT_MODEL
        else:  # anthropic (default)
            self._init_anthropic(api_key or ANTHROPIC_API_KEY)
            self.recon_model = RECON_MODEL
            self.extract_model = EXTRACT_MODEL

        self.default_model = self.recon_model

    def _init_anthropic(self, api_key: str):
        """Initialize Anthropic client"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        if not api_key:
            raise ValueError("No Anthropic API key provided. Set ANTHROPIC_API_KEY or pass api_key.")
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)

    def _init_google(self, api_key: str):
        """Initialize Google Gemini client"""
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")
        if not api_key:
            raise ValueError("No Google API key provided. Set GOOGLE_API_KEY or pass google_api_key.")
        self.google_client = genai.Client(api_key=api_key)
        self.google_api_key = api_key

    def set_provider(self, provider: str, api_key: str = None):
        """Switch to a different provider"""
        self.provider = provider
        if provider == "google":
            self._init_google(api_key or GOOGLE_API_KEY)
            self.recon_model = GEMINI_RECON_MODEL
            self.extract_model = GEMINI_EXTRACT_MODEL
        else:
            self._init_anthropic(api_key or ANTHROPIC_API_KEY)
            self.recon_model = RECON_MODEL
            self.extract_model = EXTRACT_MODEL

    def set_models(self, recon_model: str = None, extract_model: str = None):
        """Set models for recon and extraction tasks"""
        if recon_model:
            self.recon_model = recon_model
        if extract_model:
            self.extract_model = extract_model

    def call(
        self,
        prompt: str,
        system_prompt: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        response_format: str = "text"  # "text" or "json"
    ) -> str:
        """
        Make a call to the configured LLM API
        """
        model = model or self.default_model
        temperature = temperature if temperature is not None else RECON_TEMPERATURE
        max_tokens = max_tokens or MAX_TOKENS_RECON

        # Route to appropriate provider
        if self.provider == "google" or model.startswith("gemini"):
            return self._call_google(prompt, system_prompt, model, temperature, max_tokens)
        else:
            return self._call_anthropic(prompt, system_prompt, model, temperature, max_tokens)

    def _call_anthropic(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Make a call to Anthropic Claude API"""
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(MAX_RETRIES):
            try:
                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": messages,
                }

                if system_prompt:
                    kwargs["system"] = system_prompt

                if temperature > 0:
                    kwargs["temperature"] = temperature

                response = self.anthropic_client.messages.create(**kwargs)
                return response.content[0].text

            except anthropic.RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

            except anthropic.APIError as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"API error: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise

    def _call_google(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Make a call to Google Gemini API"""
        for attempt in range(MAX_RETRIES):
            try:
                # Build config
                config = types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    system_instruction=system_prompt if system_prompt else None,
                )

                # Generate response using client
                response = self.google_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )

                # Handle response
                if response.text:
                    return response.text
                elif response.candidates:
                    # Try to get text from candidates
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            return candidate.content.parts[0].text

                # If we got here, no text was returned
                return json.dumps({"error": "Empty response from Gemini"})

            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "quota" in error_str:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (2 ** attempt)
                        print(f"Rate limited. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
                elif attempt < MAX_RETRIES - 1:
                    print(f"API error: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise

    def call_recon(
        self,
        prompt: str,
        system_prompt: str = None
    ) -> str:
        """
        Call for reconnaissance tasks (scanning papers)
        Uses configured recon_model (can be cheaper like Haiku or Flash-Lite)
        """
        return self.call(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.recon_model,
            temperature=RECON_TEMPERATURE,
            max_tokens=MAX_TOKENS_RECON
        )

    def call_extract(
        self,
        prompt: str,
        system_prompt: str = None
    ) -> str:
        """
        Call for extraction tasks (needs accuracy)
        Uses configured extract_model (typically Sonnet or Gemini Flash/Pro)
        """
        return self.call(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.extract_model,
            temperature=EXTRACT_TEMPERATURE,
            max_tokens=MAX_TOKENS_EXTRACT
        )

    def call_vision(
        self,
        prompt: str,
        image_data: bytes,
        image_format: str = "png",
        system_prompt: str = None,
        model: str = None
    ) -> str:
        """
        Call with image input for vision-based extraction (charts, figures)

        Args:
            prompt: Text prompt describing what to extract
            image_data: Raw image bytes (PNG, JPEG, etc.)
            image_format: Image format - "png", "jpeg", "gif", "webp"
            system_prompt: Optional system prompt
            model: Optional model override (defaults to extract_model)

        Returns:
            LLM response text
        """
        import base64

        model = model or self.extract_model
        image_b64 = base64.standard_b64encode(image_data).decode('utf-8')

        # Map format to MIME type
        mime_types = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp"
        }
        mime_type = mime_types.get(image_format.lower(), "image/png")

        # Route to appropriate provider
        if self.provider == "google" or model.startswith("gemini"):
            return self._call_google_vision(prompt, image_b64, mime_type, system_prompt, model)
        else:
            return self._call_anthropic_vision(prompt, image_b64, mime_type, system_prompt, model)

    def _call_anthropic_vision(
        self,
        prompt: str,
        image_b64: str,
        mime_type: str,
        system_prompt: str,
        model: str
    ) -> str:
        """Make a vision call to Anthropic Claude API"""
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": image_b64
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]

        for attempt in range(MAX_RETRIES):
            try:
                kwargs = {
                    "model": model,
                    "max_tokens": MAX_TOKENS_EXTRACT,
                    "messages": messages,
                }

                if system_prompt:
                    kwargs["system"] = system_prompt

                kwargs["temperature"] = EXTRACT_TEMPERATURE

                response = self.anthropic_client.messages.create(**kwargs)
                return response.content[0].text

            except anthropic.RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

            except anthropic.APIError as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"API error: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise

    def _call_google_vision(
        self,
        prompt: str,
        image_b64: str,
        mime_type: str,
        system_prompt: str,
        model: str
    ) -> str:
        """Make a vision call to Google Gemini API"""
        import base64

        for attempt in range(MAX_RETRIES):
            try:
                # Build config
                config = types.GenerateContentConfig(
                    max_output_tokens=MAX_TOKENS_EXTRACT,
                    temperature=EXTRACT_TEMPERATURE,
                    system_instruction=system_prompt if system_prompt else None,
                )

                # Create content parts with image and text
                contents = [
                    types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type=mime_type),
                    prompt
                ]

                # Generate response using client
                response = self.google_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                # Handle response
                if response.text:
                    return response.text
                elif response.candidates:
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            return candidate.content.parts[0].text

                return json.dumps({"error": "Empty response from Gemini vision"})

            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "quota" in error_str:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (2 ** attempt)
                        print(f"Rate limited. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
                elif attempt < MAX_RETRIES - 1:
                    print(f"API error: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise

    # Aliases for backward compatibility
    def call_worker(self, prompt: str, system_prompt: str = None, expect_json: bool = False) -> str:
        """Alias for call_recon (backward compatibility)"""
        return self.call_recon(prompt, system_prompt)

    def call_orchestrator(self, prompt: str, system_prompt: str = None) -> str:
        """Alias for call_extract (backward compatibility)"""
        return self.call_extract(prompt, system_prompt)

    def call_for_json(
        self,
        prompt: str,
        system_prompt: str = None,
        use_extract_model: bool = False
    ) -> Dict[str, Any]:
        """
        Call expecting JSON response
        """
        if use_extract_model:
            response = self.call_extract(prompt, system_prompt)
        else:
            response = self.call_recon(prompt, system_prompt)

        return self.parse_json_response(response)

    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response

        Handles common issues like markdown code blocks, extra text, etc.
        Robust handling for Gemini's tendency to wrap JSON in markdown.
        """
        text = response.strip()

        # Strategy 0: Aggressively strip markdown code fences first
        # This handles cases where the JSON is wrapped in ```json ... ```
        cleaned = text

        # Remove opening code fence with language identifier
        if cleaned.startswith('```'):
            # Find the end of the first line (the language identifier line)
            first_newline = cleaned.find('\n')
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]

        # Remove closing code fence
        if cleaned.rstrip().endswith('```'):
            last_fence = cleaned.rfind('```')
            if last_fence > 0:
                cleaned = cleaned[:last_fence]

        # Try direct parse on cleaned text
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 1: Extract JSON from markdown code blocks using regex
        # Use greedy matching (.*) instead of non-greedy (.*?) for large JSON
        code_block_patterns = [
            r'```json\s*\n(.*)\n```',  # ```json ... ``` (greedy)
            r'```json\s*\n(.*)```',    # ```json ... ``` (no trailing newline)
            r'```JSON\s*\n(.*)```',    # ```JSON ... ```
            r'```\s*\n(\{.*)```',      # ``` { ... } ```
        ]

        for pattern in code_block_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        # Strategy 2: Find JSON object by locating { and matching }
        # More robust brace matching with proper string handling
        start_idx = text.find('{')
        if start_idx != -1:
            depth = 0
            in_string = False
            escape_next = False
            last_valid_end = -1

            for i, char in enumerate(text[start_idx:], start_idx):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\' and in_string:
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            last_valid_end = i
                            try:
                                return json.loads(text[start_idx:i + 1])
                            except json.JSONDecodeError:
                                # Continue to see if there's more valid JSON
                                continue

        # Strategy 3: Simple brace finding (fallback)
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_text = text[start_idx:end_idx + 1]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                # Strategy 4: Try to repair truncated JSON
                # If JSON is truncated, try to close open brackets/braces
                repaired = LLMClient._try_repair_json(json_text)
                if repaired:
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        pass

        # Strategy 5: Try to find JSON array in response
        start_idx = text.find('[')
        end_idx = text.rfind(']')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                result = json.loads(text[start_idx:end_idx + 1])
                return {"items": result} if isinstance(result, list) else result
            except json.JSONDecodeError:
                pass

        # Return error dict if all parsing fails
        return {
            "error": "Failed to parse JSON",
            "raw_response": response[:2000]  # More chars for debugging
        }

    @staticmethod
    def _try_repair_json(json_text: str) -> Optional[str]:
        """
        Try to repair truncated JSON by closing open brackets and braces.
        """
        # Count open brackets and braces
        open_braces = 0
        open_brackets = 0
        in_string = False
        escape_next = False

        for char in json_text:
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    open_braces += 1
                elif char == '}':
                    open_braces -= 1
                elif char == '[':
                    open_brackets += 1
                elif char == ']':
                    open_brackets -= 1

        # If we have unclosed brackets/braces, try to close them
        if open_braces > 0 or open_brackets > 0:
            # Remove any trailing incomplete values (like "key": or trailing comma)
            repaired = json_text.rstrip()

            # Remove trailing incomplete key-value pairs
            if repaired.endswith(','):
                repaired = repaired[:-1]
            elif repaired.endswith(':'):
                # Remove the incomplete key
                last_quote = repaired.rfind('"')
                if last_quote > 0:
                    second_last_quote = repaired.rfind('"', 0, last_quote)
                    if second_last_quote >= 0:
                        repaired = repaired[:second_last_quote].rstrip()
                        if repaired.endswith(','):
                            repaired = repaired[:-1]

            # Close any unclosed strings
            if in_string:
                repaired += '"'

            # Add closing brackets and braces
            repaired += ']' * open_brackets
            repaired += '}' * open_braces

            return repaired

        return None

    def extract_text_from_response(self, response: str, start_marker: str = None, end_marker: str = None) -> str:
        """
        Extract specific section from response using markers
        """
        text = response

        if start_marker:
            start_idx = text.find(start_marker)
            if start_idx != -1:
                text = text[start_idx + len(start_marker):]

        if end_marker:
            end_idx = text.find(end_marker)
            if end_idx != -1:
                text = text[:end_idx]

        return text.strip()

    def get_provider_info(self) -> Dict[str, str]:
        """Get information about current provider and models"""
        return {
            "provider": self.provider,
            "recon_model": self.recon_model,
            "extract_model": self.extract_model,
        }


def create_llm_client(api_key: str = None, provider: str = None, google_api_key: str = None) -> LLMClient:
    """Factory function to create LLM client"""
    return LLMClient(api_key=api_key, provider=provider, google_api_key=google_api_key)
