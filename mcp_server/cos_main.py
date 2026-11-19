#!/usr/bin/env python3
"""
Entry point for the cos deployment. Registers extra providers, then starts the stock server:

- llm.provider "codex_cli": CodexCLIClient, which runs `codex exec` on a ChatGPT subscription
- llm.provider "claude_cli": ClaudeCLIClient, which runs `claude -p` on a Claude subscription
- embedder.provider "sentence_transformers": a local CPU embedder
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from cos_providers.claude_cli_client import ClaudeCLIClient
from cos_providers.codex_cli_client import CodexCLIClient
from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

from services import factories

stock_llm_create = factories.LLMClientFactory.create
stock_embedder_create = factories.EmbedderFactory.create


def llm_create(config):
    match config.provider.lower():
        case 'codex_cli':
            return CodexCLIClient(
                config=CoreLLMConfig(
                    model=os.environ.get('CODEX_MODEL') or None,
                    small_model=os.environ.get('CODEX_SMALL_MODEL') or None,
                    max_tokens=config.max_tokens,
                )
            )
        case 'claude_cli':
            return ClaudeCLIClient(
                config=CoreLLMConfig(
                    model=os.environ.get('CLAUDE_MODEL', 'sonnet'),
                    small_model=os.environ.get('CLAUDE_SMALL_MODEL', 'haiku'),
                    max_tokens=config.max_tokens,
                )
            )
        case _:
            return stock_llm_create(config)


def embedder_create(config):
    if config.provider.lower() != 'sentence_transformers':
        return stock_embedder_create(config)
    from cos_providers.sentence_transformers_embedder import (
        SentenceTransformersEmbedder,
        SentenceTransformersEmbedderConfig,
    )

    return SentenceTransformersEmbedder(
        SentenceTransformersEmbedderConfig(
            embedding_model=config.model, embedding_dim=config.dimensions
        )
    )


factories.LLMClientFactory.create = staticmethod(llm_create)
factories.EmbedderFactory.create = staticmethod(embedder_create)

if __name__ == '__main__':
    from graphiti_mcp_server import main

    main()
