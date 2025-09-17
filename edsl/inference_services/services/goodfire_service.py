import aiohttp
import json
import requests
import os
from typing import Any, List, Optional, TYPE_CHECKING

import openai
from edsl.inference_services.inference_service_abc import InferenceServiceABC
from edsl.inference_services.decorators import report_errors_async
from edsl.inference_services.services.message_builder import MessageBuilder

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from edsl.language_models import LanguageModel
    from edsl.scenarios.file_store import FileStore as Files
    from edsl.invigilators.invigilator_base import InvigilatorBase as InvigilatorAI


class GoodfireService(InferenceServiceABC):
    """Goodfire service class."""

    _inference_service_ = "goodfire"
    _env_key_name_ = "GOODFIRE_API_KEY"
    _base_url_ = "https://api.goodfire.ai/api/inference/v1"
    _chat_completions_url_ = f"{_base_url_}/chat/completions"
    _models_list_cache: List[str] = []

    _sync_client_instances = {}
    _async_client_instances = {}

    # Required attributes for InferenceServiceABC
    key_sequence = ["choices", 0, "message", "content"]
    usage_sequence = ["usage"]
    input_token_name = "prompt_tokens"
    output_token_name = "completion_tokens"
    available_models_url = "https://api.goodfire.ai/api/inference/v1/models"

    @classmethod
    def get_model_info(cls, api_key=None):
        """Get raw model info without wrapping in ModelInfo."""
        if api_key is None:
            api_key = os.getenv(cls._env_key_name_)
        if api_key is None:
            raise ValueError(f"API key for {cls._inference_service_} is not set")

        # For now, return a hardcoded list since Goodfire API might not have a models endpoint
        return [
            {
                "id": "meta-llama/Llama-3.3-70B-Instruct",
                "object": "model",
                "created": 1234567890,
                "owned_by": "goodfire",
            }
        ]

    @classmethod
    def available(cls) -> List[str]:
        return [
            "meta-llama/Llama-3.3-70B-Instruct",
        ]

    @classmethod
    def create_model(cls, model_name, model_class_name=None) -> "LanguageModel":
        if model_class_name is None:
            model_class_name = cls.to_class_name(model_name)

        # Import LanguageModel only when actually creating a model
        from edsl.language_models import LanguageModel

        class LLM(LanguageModel):
            """
            Child class of LanguageModel for interacting with Goodfire models
            """

            key_sequence = cls.key_sequence
            usage_sequence = cls.usage_sequence
            input_token_name = cls.input_token_name
            output_token_name = cls.output_token_name

            _inference_service_ = cls._inference_service_
            _model_ = model_name

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            _parameters_ = {
                "temperature": 0.5,
                "max_tokens": 1000,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "logprobs": False,
                "top_logprobs": 3,
            }

            def sync_client(self):
                return cls.sync_client(api_key=self.api_token)

            def async_client(self):
                return cls.async_client(api_key=self.api_token)

            @classmethod
            def available(cls) -> list[str]:
                return cls.sync_client().models.list()

            @report_errors_async
            async def async_execute_model_call(
                self,
                user_prompt: str,
                system_prompt: str = "",
                question_name: Optional[str] = None,
                files_list: Optional[List["Files"]] = None,
                invigilator: Optional["InvigilatorAI"] = None,
                controller: Optional[dict] = None,
            ) -> dict[str, Any]:
                """Calls the Goodfire API and returns the API response."""

                # Use MessageBuilder to construct messages
                message_builder = MessageBuilder(
                    model=self.model,
                    files_list=files_list,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    omit_system_prompt_if_empty=self.omit_system_prompt_if_empty,
                )

                client = self.async_client()
                messages = message_builder.get_messages(sync_client=self.sync_client())

                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                    "logprobs": self.logprobs,
                    "top_logprobs": self.top_logprobs if self.logprobs else None,
                }

                # Add controller parameters if provided
                controller_param = controller or self.parameters.get("controller")
                if controller_param:
                    params.update({"extra_body": {"controller": controller_param}})

                try:
                    response = await client.chat.completions.create(**params)
                    return response.model_dump()
                except Exception as e:
                    print(e)
                    raise

        # Ensure the class name is "LanguageModel" for proper serialization
        LLM.__name__ = "LanguageModel"
        LLM.__qualname__ = "LanguageModel"

        return LLM

    @classmethod
    def sync_client(cls, api_key=None):
        if api_key is None:
            api_key = os.getenv(cls._env_key_name_)
        if api_key not in cls._sync_client_instances:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=cls._base_url_,
            )
            cls._sync_client_instances[api_key] = client
        return cls._sync_client_instances[api_key]

    @classmethod
    def async_client(cls, api_key=None):
        if api_key is None:
            api_key = os.getenv(cls._env_key_name_)
        if api_key not in cls._async_client_instances:
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=cls._base_url_,
            )
            cls._async_client_instances[api_key] = client
        return cls._async_client_instances[api_key]
