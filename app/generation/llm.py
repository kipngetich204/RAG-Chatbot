import requests


class OllamaLLM:

    def __init__(
        self,
        model_name,
        base_url="http://localhost:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
        )

        data = response.json()

        return data["response"]