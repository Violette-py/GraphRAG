# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A text-completion based LLM."""

import logging

from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
)
from .qwen_client import QwenClient
from .qwen_configuration import QwenConfiguration

from .utils import get_completion_llm_args

log = logging.getLogger(__name__)

class QwenCompletionLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A text-completion based LLM for Qwen."""

    _client: QwenClient
    _configuration: QwenConfiguration

    def __init__(self, client: QwenClient, configuration: QwenConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput | None:
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self.configuration
        )
        completion = await self.client.create_completion(prompt=input, **args)
        return completion.choices[0].text
