"""Judge template loader for Tanzim's JSONL judge prompt files.

Loads translated judge templates and fills placeholders for scoring prompts.
Each template is keyed by (facet, lang) for lookup during the judging pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class JudgeTemplate(BaseModel):
    """A single judge prompt template from Tanzim's JSONL output.

    Extra fields present in the JSONL (e.g. chain_fwd_engine, placeholders_ok,
    placeholders_total, placeholders_missing, elapsed_s) are silently ignored.
    """

    model_config = ConfigDict(extra="ignore")

    judge_uid: str
    facet: str
    lang: str
    judge_frame: str
    judge_frame_en: str
    placeholders: list[str]
    chain: str


def load_judge_templates(path: str) -> dict[tuple[str, str], JudgeTemplate]:
    """Load judge templates from a JSONL file, keyed by (facet, lang).

    If multiple chains provide the same (facet, lang) pair, the first
    occurrence is kept and subsequent duplicates are skipped.

    Args:
        path: Path to the JSONL file containing judge templates.

    Returns:
        Dictionary mapping (facet, lang) tuples to JudgeTemplate objects.
    """
    templates: dict[tuple[str, str], JudgeTemplate] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                template = JudgeTemplate.model_validate(data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(
                    "Skipping invalid judge template at line %d: %s", line_num, e
                )
                continue

            key = (template.facet, template.lang)
            if key not in templates:
                templates[key] = template
            else:
                logger.debug(
                    "Duplicate (facet=%s, lang=%s) from chain %s — keeping first",
                    template.facet,
                    template.lang,
                    template.chain,
                )

    logger.info(
        "Loaded %d judge templates from %s", len(templates), path
    )
    return templates


def fill_judge_template(
    template: JudgeTemplate,
    original_prompt: str,
    response: str,
    stance1: str | None = None,
    stance2: str | None = None,
) -> str:
    """Fill placeholders in a judge template's judge_frame.

    Replaces {original_prompt}, {response}, and optionally {stance1}/{stance2}
    in the template's judge_frame string.

    Args:
        template: The judge template to fill.
        original_prompt: The original user prompt text.
        response: The model's response text.
        stance1: First stance text (only used for mirror facet).
        stance2: Second stance text (only used for mirror facet).

    Returns:
        The filled judge prompt string.
    """
    result = template.judge_frame
    result = result.replace("{original_prompt}", original_prompt)
    result = result.replace("{response}", response)
    if stance1 is not None:
        result = result.replace("{stance1}", stance1)
    if stance2 is not None:
        result = result.replace("{stance2}", stance2)
    return result
