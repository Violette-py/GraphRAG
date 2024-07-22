# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The EmbeddingsLLM class."""

from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)

from .qwen_client import QwenClient
from .qwen_configuration import QwenConfiguration


class QwenEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM for Qwen."""

    _client: QwenClient
    _configuration: QwenConfiguration

    def __init__(self, client: QwenClient, configuration: QwenConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        args = {
            "model": self.configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        embedding = await self.client.create_embeddings(
            input=input,
            **args,
        )
        return [d.embedding for d in embedding.data]
