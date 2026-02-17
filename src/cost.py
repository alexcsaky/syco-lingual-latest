"""Cost estimation for API calls."""

from __future__ import annotations


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_rate: float,
    output_rate: float,
) -> float:
    """Estimate USD cost given token counts and per-million-token rates."""
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
