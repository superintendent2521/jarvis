import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

# Load environment variables from .env file
load_dotenv()


class OpenRouterClient:
    """Client for interacting with OpenRouter API using OpenAI SDK."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )
        self.default_model = os.getenv("DEFAULT_MODEL", "openai/gpt-4o")  # Default model

    def set_model(self, model: str):
        """Set the default model to use."""
        self.default_model = model

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> ChatCompletionMessage:
        """
        Make a chat completion request to OpenRouter.

        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to self.default_model)
            tools: List of tool schemas for function calling
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            ChatCompletionMessage object
        """
        model = model or self.default_model

        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            request_params["max_tokens"] = max_tokens

        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"

        if stream:
            # For streaming, we'd need to handle the stream differently
            # For now, we'll keep it simple and not implement streaming
            pass

        try:
            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}")

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter."""
        try:
            models = self.client.models.list()
            return [{"id": model.id, "name": model.id} for model in models.data]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
