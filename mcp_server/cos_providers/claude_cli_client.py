"""
LLM client that runs the `claude` CLI (`claude -p`) as a subprocess, so model calls use a
Claude subscription login instead of an API key.
"""

import asyncio
import json
import logging
import os
import tempfile
import typing

from pydantic import BaseModel, ValidationError

from graphiti_core.prompts.models import Message
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import EmptyResponseError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'sonnet'
DEFAULT_SMALL_MODEL = 'haiku'
DEFAULT_TIMEOUT_SECONDS = 300


class ClaudeCLIClient(LLMClient):
    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        claude_bin: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        if config is None:
            config = LLMConfig(model=DEFAULT_MODEL, small_model=DEFAULT_SMALL_MODEL)
        super().__init__(config, cache)
        self.claude_bin = claude_bin or os.environ.get('CLAUDE_BIN', 'claude')
        self.timeout = timeout
        self.workdir = tempfile.mkdtemp(prefix='graphiti-claude-')

    def _model_for(self, model_size: ModelSize, model: str | None) -> str:
        if model:
            return model
        if model_size == ModelSize.small:
            return self.small_model or DEFAULT_SMALL_MODEL
        return self.model or DEFAULT_MODEL

    def _build_prompt(
        self, messages: list[Message], response_model: type[BaseModel] | None
    ) -> tuple[str, str]:
        system_parts: list[str] = []
        user_parts: list[str] = []
        for m in messages:
            content = self._clean_input(m.content)
            if m.role == 'system':
                system_parts.append(content)
            else:
                user_parts.append(content)
        if response_model is not None:
            schema = json.dumps(response_model.model_json_schema())
            system_parts.append(
                'Respond with a single JSON object that validates against this JSON schema. '
                'Output only the JSON, with no prose and no code fences.\n\n' + schema
            )
        return '\n\n'.join(system_parts), '\n\n'.join(user_parts)

    async def _run_claude(self, system_prompt: str, prompt: str, model: str) -> str:
        args = [
            self.claude_bin,
            '-p',
            '--model',
            model,
            '--output-format',
            'json',
            '--tools',
            '',
            '--strict-mcp-config',
            '--mcp-config',
            '{"mcpServers":{}}',
            '--setting-sources',
            '',
            '--no-session-persistence',
            '--system-prompt',
            system_prompt,
        ]
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workdir,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode()), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        if proc.returncode != 0:
            raise RuntimeError(
                f'claude exited with {proc.returncode}: {stderr.decode(errors="replace")[:500]}'
            )
        envelope = json.loads(stdout)
        if envelope.get('is_error'):
            raise RuntimeError(f'claude returned an error: {envelope.get("result")}')
        result = envelope.get('result') or ''
        if not result:
            raise EmptyResponseError('claude returned an empty response')
        return result

    @staticmethod
    def _extract_json(text: str) -> dict[str, typing.Any]:
        text = text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else ''
            text = text.rsplit('```', 1)[0]
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            raise ValueError('no JSON object in claude response')
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
        system_prompt, prompt = self._build_prompt(messages, response_model)
        chosen_model = self._model_for(model_size, model)
        last_error: Exception | None = None
        for attempt in range(2):
            text = await self._run_claude(system_prompt, prompt, chosen_model)
            try:
                data = self._extract_json(text)
                if response_model is not None:
                    response_model.model_validate(data)
                return data
            except (ValueError, ValidationError) as e:
                last_error = e
                logger.warning(
                    'claude-cli parse failure (attempt %d, model %s, schema %s): %s',
                    attempt + 1,
                    chosen_model,
                    response_model.__name__ if response_model else None,
                    e,
                )
        raise ValueError(f'claude returned invalid JSON twice: {last_error}')
