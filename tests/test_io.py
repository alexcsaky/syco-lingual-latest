# tests/test_io.py
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.io import JsonlWriter, load_completed_keys, load_jsonl
from src.schemas import ModelResponse


def _make_response(prompt_id: str, model: str) -> ModelResponse:
    return ModelResponse(
        prompt_id=prompt_id, item_id=prompt_id.rsplit("_", 1)[0],
        facet="mirroring", variant="a", language="en",
        prompt_text="test", model=model, model_version="v1",
        response_text="test response", response_tokens=10,
        reasoning_tokens=0, finish_reason="stop",
        detected_language="en", language_match=True,
        timestamp=datetime.now(timezone.utc), latency_ms=100,
        run_id="test", estimated_cost_usd=0.001,
    )


class TestJsonlWriter:
    @pytest.mark.asyncio
    async def test_write_and_read_back(self, tmp_path):
        path = tmp_path / "test.jsonl"
        writer = JsonlWriter(str(path))
        r = _make_response("mirror_001_a", "gpt-5")
        await writer.write(r)
        writer.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["prompt_id"] == "mirror_001_a"

    @pytest.mark.asyncio
    async def test_append_mode(self, tmp_path):
        path = tmp_path / "test.jsonl"
        writer = JsonlWriter(str(path))
        await writer.write(_make_response("mirror_001_a", "gpt-5"))
        await writer.write(_make_response("mirror_001_b", "gpt-5"))
        writer.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, tmp_path):
        path = tmp_path / "test.jsonl"
        writer = JsonlWriter(str(path))

        async def write_batch(start: int):
            for i in range(start, start + 20):
                await writer.write(_make_response(f"item_{i:03d}_a", "gpt-5"))

        await asyncio.gather(write_batch(0), write_batch(20), write_batch(40))
        writer.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 60
        # Verify each line is valid JSON
        for line in lines:
            json.loads(line)


class TestLoadCompletedKeys:
    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        keys = load_completed_keys(str(path), key_fields=["prompt_id", "model"])
        assert keys == set()

    def test_nonexistent_file(self, tmp_path):
        path = tmp_path / "nope.jsonl"
        keys = load_completed_keys(str(path), key_fields=["prompt_id", "model"])
        assert keys == set()

    def test_loads_correct_keys(self, tmp_path):
        path = tmp_path / "test.jsonl"
        r1 = _make_response("mirror_001_a", "gpt-5")
        r2 = _make_response("mirror_001_a", "claude-sonnet-4-5")
        path.write_text(r1.model_dump_json() + "\n" + r2.model_dump_json() + "\n")

        keys = load_completed_keys(str(path), key_fields=["prompt_id", "model"])
        assert ("mirror_001_a", "gpt-5") in keys
        assert ("mirror_001_a", "claude-sonnet-4-5") in keys
        assert len(keys) == 2

    def test_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        r1 = _make_response("mirror_001_a", "gpt-5")
        path.write_text(r1.model_dump_json() + "\n" + "CORRUPT LINE\n")

        keys = load_completed_keys(str(path), key_fields=["prompt_id", "model"])
        assert len(keys) == 1


class TestLoadJsonl:
    def test_load_typed(self, tmp_path):
        path = tmp_path / "test.jsonl"
        r = _make_response("mirror_001_a", "gpt-5")
        path.write_text(r.model_dump_json() + "\n")

        results = load_jsonl(str(path), ModelResponse)
        assert len(results) == 1
        assert results[0].prompt_id == "mirror_001_a"
