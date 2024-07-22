import pytest
from aioresponses import aioresponses

from graphrag.llm.qwen import create_qwen_client


@pytest.fixture
def config_file():
    return "config.ini"

@pytest.fixture
def client(config_file):
    return create_qwen_client(config_file)

@pytest.mark.asyncio
@aioresponses()
async def test_create_embeddings(mocked, client):
    url = f"{client.base_url}/text-embedding/text-embedding"
    mocked.post(url, payload={"embeddings": [1, 2, 3]})

    input_data = ["test"]
    model = "test_model"
    response = await client.create_embeddings(input_data, model)

    assert response == {"embeddings": [1, 2, 3]}

@pytest.mark.asyncio
@aioresponses()
async def test_create_completion(mocked, client):
    url = f"{client.base_url}/aigc/text-generation/generation"
    mocked.post(url, payload={"completion": "completed text"})

    prompt = "test prompt"
    model = "test_model"
    response = await client.create_completion(prompt, model)

    assert response == {"completion": "completed text"}

@pytest.mark.asyncio
@aioresponses()
async def test_create_chat_completion(mocked, client):
    url = f"{client.base_url}/aigc/text-generation/generation"
    mocked.post(url, payload={"chat_completion": "chat response"})

    messages = [{"role": "user", "content": "hello"}]
    model = "test_model"
    response = await client.create_chat_completion(messages, model)

    assert response == {"chat_completion": "chat response"}
