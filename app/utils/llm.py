import requests, os
from litellm import completion

class LLM:
    def __init__(self, api_key):
        os.environ["COHERE_API_KEY"] = api_key

    def generate_response(self, prompt, temperature=0.5, max_tokens=1000):
        response = completion(
            model="command-r-plus",
            messages=[{"content": prompt, "role": "user"}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
