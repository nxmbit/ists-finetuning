import asyncio
import os
from asyncio import Semaphore
from enum import Enum
import json

from aiohttp import ClientSession


class ErrorCode(Enum):
    TIMEOUT = (0, "Timeout error")
    PROCESSING = (1, "Processing error")
    NO_CONTENT = (2, "No input content error")
    HTTP_CONNECTION = (3, "HTTP connection error")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

    def __str__(self):
        return f"[{self.code}] {self.description}"


class VLLMBackend:
    URL = "http://127.0.0.1:8000/v1/chat/completions"
    HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    TIMEOUT = 180

    def __init__(
        self,
        base_model: str,
        system_prompt: str,
        temperature: float,
        output_path: str,
    ):
        self.base_model = base_model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.output_path = output_path

    async def __call__(
        self,
        prompt: str,
        idx: int,
        semaphore: Semaphore,
        session: ClientSession,
    ):
        payload = self._build_payload(prompt)
        async with semaphore:
            try:
                data = await self._make_api_call(session, payload)
            except asyncio.TimeoutError:
                print(f"Timeout while processing {idx}")
                self._save(
                    input=prompt,
                    result=ErrorCode.TIMEOUT,
                    idx=idx,
                )
                return
            except Exception as e:
                print(f"Exception while processing {idx}: {e}")
                self._save(
                    input=prompt,
                    result=ErrorCode.PROCESSING,
                    idx=idx,
                )
                return
            if "error" in data:
                print(f"Failed to process {idx}: {data['error']}")
                self._save(
                    input=prompt,
                    result=ErrorCode.HTTP_CONNECTION,
                    idx=idx,
                )
                return
            result = data.get("choices", [{}])[0].get("message", {}).get("content")
            if not result:
                print(f"No content in response for {idx}")
                self._save(
                    input=prompt,
                    result=ErrorCode.NO_CONTENT,
                    idx=idx,
                )
                return
            self._save(prompt, result, idx)

            return {
                "idx": idx,
                "prompt": prompt,
                "result": result,
            }

    def _build_payload(
        self,
        prompt: str,
    ) -> dict:
        return {
            "model": self.base_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
        }

    async def _make_api_call(
        self,
        session: ClientSession,
        payload: dict,
    ) -> dict:
        async with session.post(
            self.URL, headers=self.HEADERS, json=payload, timeout=self.TIMEOUT
        ) as response:
            if response.status != 200:
                return {"error": f"HTTP {response.status}"}
            data = await response.json()
            return data

    def _save(
        self,
        input: str,
        result: str | ErrorCode,
        idx: int,
    ):
        if isinstance(result, ErrorCode):
            result = str(result)

        os.makedirs(self.output_path, exist_ok=True)
        output_filepath = os.path.join(self.output_path, f"{idx}.json")

        payload = {
            "index": idx,
            "input": input,
            "result": result,
        }

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
