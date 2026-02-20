"""Abstract base class for all LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.schemas import ProviderResponse


class BaseProvider(ABC):
    def __init__(self, family: str, model_id: str):
        self.family = family
        self.model_id = model_id

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse: ...

    @abstractmethod
    async def complete_structured(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse: ...

    async def close(self) -> None:
        """Close underlying resources. No-op by default."""
        pass
