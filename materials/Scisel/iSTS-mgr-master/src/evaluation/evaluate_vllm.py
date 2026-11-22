import asyncio
from src.utils import select_label_from_response
from asyncio import Semaphore
from typing import Any, Iterable
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
from src.backend.vllm import VLLMBackend


class VllmEvaluator:
    def __init__(
        self,
        model: str,
        system_prompt: str,
        output_dir: str,
        temperature: float = 0.1,
        concurrency: int = 32,
    ):
        self.system_prompt = system_prompt
        self.concurrency = concurrency
        self.llm_backend = VLLMBackend(
            base_model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            output_path=output_dir,
        )

    async def _call_backend(self, prompts: list[str]):
        semaphore = Semaphore(self.concurrency)
        async with ClientSession() as session:
            tasks = [
                self.llm_backend(
                    prompt=row,
                    idx=idx,
                    semaphore=semaphore,
                    session=session,
                )
                for idx, row in enumerate(prompts)
            ]
            raw_results = await tqdm_asyncio(
                smoothing=0.05
            ).gather(*tasks)

        results = [r["result"] if r is not None else None for r in raw_results]

        return results

    def _prepare_input(self, input: Iterable[Any]) -> list[dict[str, str]]:
        
        return input

    def _process_model_input(self, input: list[str]) -> list[str]:
        return input

    def _process_model_output(self, output: list[str]) -> list[str]:
        return output

    def _prepare_message(
        self, system_prompt: str, user_instruction: str, model_generation: str
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_instruction},
            {"role": "assistant", "content": model_generation},
        ]

    def _process_messages(
        self,
        system_prompt: str,
        user_instructions: list[str],
        model_generations: list[str],
    ) -> list[list[dict[str, str]]]:
        assert len(user_instructions) == len(model_generations), (
            "Instructions count do not match count of generated outputs"
        )
        messages = [
            self._prepare_message(system_prompt, user_instruction, model_generation)
            for user_instruction, model_generation in zip(
                user_instructions, model_generations
            )
        ]

        return messages

    def __call__(self, inputs: list[str], *args, **kwds):
        inputs = self._prepare_input(inputs)
        prompts = self._process_model_input(inputs)
        outputs = asyncio.run(self._call_backend(prompts))
        return self._process_model_output(outputs)
    
