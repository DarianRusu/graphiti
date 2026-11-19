"""
LLM client that runs the `codex` CLI (`codex exec`) as a subprocess, so model calls use a
ChatGPT subscription login instead of an API key.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import typing
from pathlib import Path

from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, ValidationError

from graphiti_core.prompts.models import Message
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import EmptyResponseError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300


class CodexCLIClient(LLMClient):
    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        codex_bin: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        super().__init__(config or LLMConfig(), cache)
        self.codex_bin = codex_bin or os.environ.get('CODEX_BIN', 'codex')
        self.timeout = timeout

    def _model_for(self, model_size: ModelSize, model: str | None) -> str | None:
        if model:
            return model
        if model_size == ModelSize.small and self.small_model:
            return self.small_model
        return self.model

    def _build_prompt(
        self, messages: list[Message], response_model: type[BaseModel] | None
    ) -> str:
        parts = [
            'Answer from the text below only. Do not run commands or read files.',
        ]
        for m in messages:
            parts.append(self._clean_input(m.content))
        if response_model is not None:
            parts.append('Reply with a single JSON object that matches the required schema.')
        return '\n\n'.join(parts)

    async def _run_codex(
        self, prompt: str, model: str | None, response_model: type[BaseModel] | None
    ) -> str:
        workdir = Path(tempfile.mkdtemp(prefix='graphiti-codex-'))
        try:
            out_file = workdir / 'answer.txt'
            args = [
                self.codex_bin,
                'exec',
                '--skip-git-repo-check',
                '--ephemeral',
                '--ignore-user-config',
                '--ignore-rules',
                '--sandbox',
                'read-only',
                '--color',
                'never',
                '-C',
                str(workdir),
                '-o',
                str(out_file),
            ]
            if model:
                args += ['-m', model]
            if response_model is not None:
                schema_file = workdir / 'schema.json'
                schema_file.write_text(json.dumps(to_strict_json_schema(response_model)))
                args += ['--output-schema', str(schema_file)]
            args.append('-')

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(prompt.encode()), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise
            if proc.returncode != 0:
                raise RuntimeError(
                    f'codex exited with {proc.returncode}: '
                    f'{stderr.decode(errors="replace")[-500:]}'
                )
            result = out_file.read_text().strip() if out_file.exists() else ''
            if not result:
                raise EmptyResponseError('codex returned an empty response')
            return result
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _extract_json(text: str) -> dict[str, typing.Any]:
        text = text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else ''
            text = text.rsplit('```', 1)[0]
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            raise ValueError('no JSON object in codex response')
        return json.loads(text[start : end + 1])

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
    ) -> dict[str, typing.Any]:
        prompt = self._build_prompt(messages, response_model)
        chosen_model = self._model_for(model_size, model)
        last_error: Exception | None = None
        for attempt in range(2):
            text = await self._run_codex(prompt, chosen_model, response_model)
            try:
                data = self._extract_json(text)
                if response_model is not None:
                    response_model.model_validate(data)
                return data
            except (ValueError, ValidationError) as e:
                last_error = e
                logger.warning(
                    'codex-cli parse failure (attempt %d, model %s, schema %s): %s',
                    attempt + 1,
                    chosen_model,
                    response_model.__name__ if response_model else None,
                    e,
                )
        raise ValueError(f'codex returned invalid JSON twice: {last_error}')
