"""
LLM client that calls the Codex backend (chatgpt.com/backend-api/codex) directly with the Codex CLI's
ChatGPT login, sending only Graphiti's own prompt. Model calls count against the ChatGPT subscription.

The access token is read from the Codex CLI's auth.json on every call. This client never refreshes it:
the Codex CLI does that whenever it runs, and the batch job runs it just before feeding Graphiti.
"""

import json
import logging
import os
import typing
from pathlib import Path

from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, ValidationError

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import EmptyResponseError
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)

CODEX_BACKEND_URL = 'https://chatgpt.com/backend-api/codex'
CLIENT_NAME = 'cos-graphiti'


class CodexBackendClient(LLMClient):
    def __init__(
        self,
        config: LLMConfig,
        cache: bool = False,
        auth_path: str | None = None,
        reasoning_effort: str | None = 'low',
    ):
        super().__init__(config, cache)
        self.auth_path = Path(auth_path or os.path.expanduser('~/.codex/auth.json'))
        self.reasoning_effort = reasoning_effort

    def _client(self) -> AsyncOpenAI:
        tokens = json.loads(self.auth_path.read_text())['tokens']
        return AsyncOpenAI(
            base_url=CODEX_BACKEND_URL,
            api_key=tokens['access_token'],
            default_headers={
                'ChatGPT-Account-ID': tokens['account_id'],
                'originator': CLIENT_NAME,
                'User-Agent': f'{CLIENT_NAME}/1.0',
            },
        )

    def _model_for(self, model_size: ModelSize, model: str | None) -> str:
        if model:
            return model
        if model_size == ModelSize.small and self.small_model:
            return self.small_model
        return self.model

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
    ) -> dict[str, typing.Any]:
        instructions = '\n\n'.join(self._clean_input(m.content) for m in messages if m.role == 'system')
        user_text = '\n\n'.join(self._clean_input(m.content) for m in messages if m.role != 'system')
        request: dict[str, typing.Any] = {
            'model': self._model_for(model_size, model),
            'instructions': instructions,
            'input': [{'role': 'user', 'content': user_text}],
            'store': False,
        }
        if response_model is not None:
            request['text'] = {
                'format': {
                    'type': 'json_schema',
                    'name': response_model.__name__,
                    'schema': to_strict_json_schema(response_model),
                    'strict': True,
                }
            }
        else:
            request['text'] = {'format': {'type': 'json_object'}}
        if self.reasoning_effort:
            request['reasoning'] = {'effort': self.reasoning_effort}

        client = self._client()
        try:
            async with client.responses.stream(**request) as stream:
                final = await stream.get_final_response()
        finally:
            await client.close()

        text = (final.output_text or '').strip()
        if not text:
            raise EmptyResponseError('codex backend returned an empty response')
        try:
            data = json.loads(text)
            if response_model is not None:
                response_model.model_validate(data)
        except (ValueError, ValidationError) as e:
            logger.warning('codex-backend parse failure (schema %s): %s', getattr(response_model, '__name__', None), e)
            raise
        return data
