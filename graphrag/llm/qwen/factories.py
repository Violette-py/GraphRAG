# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Factory functions for creating Qwen LLMs."""

import asyncio

from graphrag.llm.base import CachingLLM, RateLimitingLLM
from graphrag.llm.limiting import LLMLimiter
from graphrag.llm.types import (
    LLM,
    CompletionLLM,
    EmbeddingLLM,
    ErrorHandlerFn,
    LLMCache,
    LLMInvocationFn,
    OnCacheActionFn,
)
from . import QwenClient

from .json_parsing_llm import JsonParsingLLM
from .qwen_chat_llm import QwenChatLLM
from .qwen_completion_llm import QwenCompletionLLM
from .qwen_configuration import QwenConfiguration
from .qwen_embeddings_llm import QwenEmbeddingsLLM
from .qwen_history_tracking_llm import QwenHistoryTrackingLLM
from .qwen_token_replacing_llm import QwenTokenReplacingLLM
from .utils import (
    RATE_LIMIT_ERRORS,
    RETRYABLE_ERRORS,
    get_completion_cache_args,
    get_sleep_time_from_error,
    get_token_counter,
)


def create_qwen_chat_llm(
    client: QwenClient,
    config: QwenConfiguration,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> CompletionLLM:
    """Create a Qwen chat LLM."""
    operation = "chat"
    result = QwenChatLLM(client, config)
    result.on_error(on_error)
    if limiter is not None or semaphore is not None:
        result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)
    if cache is not None:
        result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)
    result = QwenHistoryTrackingLLM(result)
    result = QwenTokenReplacingLLM(result)
    return JsonParsingLLM(result)


def create_qwen_completion_llm(
    client: QwenClient,
    config: QwenConfiguration,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> CompletionLLM:
    """Create a Qwen completion LLM."""
    operation = "completion"
    result = QwenCompletionLLM(client, config)
    result.on_error(on_error)
    if limiter is not None or semaphore is not None:
        result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)
    if cache is not None:
        result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)
    return QwenTokenReplacingLLM(result)


def create_qwen_embedding_llm(
    client: QwenClient,
    config: QwenConfiguration,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> EmbeddingLLM:
    """Create a Qwen embeddings LLM."""
    operation = "embedding"
    result = QwenEmbeddingsLLM(client, config)
    result.on_error(on_error)
    if limiter is not None or semaphore is not None:
        result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)
    if cache is not None:
        result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)
    return result


def _rate_limited(
    delegate: LLM,
    config: QwenConfiguration,
    operation: str,
    limiter: LLMLimiter | None,
    semaphore: asyncio.Semaphore | None,
    on_invoke: LLMInvocationFn | None,
):
    result = RateLimitingLLM(
        delegate,
        config,
        operation,
        RETRYABLE_ERRORS,
        RATE_LIMIT_ERRORS,
        limiter,
        semaphore,
        get_token_counter(config),
        get_sleep_time_from_error,
    )
    result.on_invoke(on_invoke)
    return result


def _cached(
    delegate: LLM,
    config: QwenConfiguration,
    operation: str,
    cache: LLMCache,
    on_cache_hit: OnCacheActionFn | None,
    on_cache_miss: OnCacheActionFn | None,
):
    cache_args = get_completion_cache_args(config)
    result = CachingLLM(delegate, cache_args, operation, cache)
    result.on_cache_hit(on_cache_hit)
    result.on_cache_miss(on_cache_miss)
    return result
