import json
from typing import Generator

import requests
from requests.exceptions import RequestException


class OllamaLLM:

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url

    def generate(
        self,
        prompt: str,
        silent: bool = False,
    ) -> str:

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=None,
            )

            response.raise_for_status()

            payload = response.json()
            #print(f"payload: {payload}")

            return payload.get("response", "")

        except RequestException as exc:
            raise RuntimeError(
                f"Ollama request failed: {exc}"
            ) from exc

    def stream_generate(
        self,
        prompt: str,
    ) -> Generator[str, None, None]:

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                },
                stream=True,
                timeout=None,
            )

            response.raise_for_status()

            for line in response.iter_lines():

                if not line:
                    continue

                payload = json.loads(
                    line.decode("utf-8")
                )

                token = payload.get(
                    "response",
                    ""
                )

                if token:
                    yield token

        except RequestException as exc:
            raise RuntimeError(
                f"Ollama streaming failed: {exc}"
            ) from exc

    def is_available(self) -> bool:

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=None,
            )

            response.raise_for_status()

            return True

        except RequestException:
            return False