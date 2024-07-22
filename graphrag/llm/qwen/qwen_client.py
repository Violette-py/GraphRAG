import aiohttp
import configparser

class QwenClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    async def create_embeddings(self, input: list[str] | str, model: str, **kwargs):
        url = f"{self.base_url}/text-embedding/text-embedding"
        data = {
            "input": input,
            "model": model
        }
        data.update(kwargs)
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    response.raise_for_status()

    async def create_completion(self, prompt: str, model: str, **kwargs):
        url = f"{self.base_url}/aigc/text-generation/generation"
        data = {
            "prompt": prompt,
            "model": model
        }
        data.update(kwargs)
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    response.raise_for_status()

    async def create_chat_completion(self, messages: list, model: str, **kwargs):
        url = f"{self.base_url}/aigc/text-generation/generation"
        data = {
            "messages": messages,
            "model": model
        }
        data.update(kwargs)
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    response.raise_for_status()

def create_qwen_client(config_file: str = 'config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    base_url = config['DEFAULT']['base_url']
    api_key = config['DEFAULT']['api_key']
    return QwenClient(base_url, api_key)
