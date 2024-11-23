import requests
import os
from dotenv import load_dotenv

load_dotenv()

class LLM:
    def __init__(self):
        self.api_key = os.getenv("AZURE_API_KEY_GPT_4_mini")
        self.endpoint = os.getenv("AZURE_ENDPOINT_GPT_4_mini")

    def generate_response(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        data = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 3000,
            "temperature": 0.5,
        }

        response = requests.post(self.endpoint, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return ValueError("Error")
