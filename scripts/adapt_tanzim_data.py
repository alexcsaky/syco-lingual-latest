#!/usr/bin/env python3
"""Adapt TanzimSyco translation output → syco-lingual evaluation pipeline format.

Reads:
  TanzimSyco/data/translated/prompts/{lang}.jsonl   (10 × 280 = 2,800 prompts)
  TanzimSyco/data/translated/judges/{lang}.json      (10 × 4 facets = 40 templates)

Writes:
  data/production/prompts.jsonl   (2,800 validated prompt records)
  data/production/judges.jsonl    (40 validated judge template records)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import TranslatedPrompt
from src.judge_templates import JudgeTemplate

# ── Paths ──────────────────────────────────────────────────────────────

TANZIM_ROOT = PROJECT_ROOT.parent / "TanzimSyco"
TANZIM_PROMPTS_DIR = TANZIM_ROOT / "data" / "translated" / "prompts"
TANZIM_JUDGES_DIR = TANZIM_ROOT / "data" / "translated" / "judges"

OUTPUT_DIR = PROJECT_ROOT / "data" / "production"

LANGUAGES = ["en", "ja", "bn", "de", "es", "fr", "zh", "ar", "sl", "lv"]


def adapt_prompts() -> list[TranslatedPrompt]:
    """Read all per-language prompt files and convert to pipeline schema."""
    prompts: list[TranslatedPrompt] = []
    errors = 0

    for lang in LANGUAGES:
        path = TANZIM_PROMPTS_DIR / f"{lang}.jsonl"
        if not path.exists():
            print(f"  WARNING: Missing {path}")
            continue

        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  ERROR: {path}:{line_num} — bad JSON: {e}")
                    errors += 1
                    continue

                # Generate prompt_uid
                item_id = raw["item_id"]
                facet = raw["facet"]
                run = raw["run"]
                prompt_uid = f"deepl_deepl__{lang}__{facet}__{item_id}__{run}"

                # Extract chrf_score from sanity_checks if available
                sanity = raw.get("sanity_checks") or {}
                chrf_score = sanity.get("chrf_score")

                # Handle fields_translated: empty dict → None for consistency
                ft = raw.get("fields_translated")
                if ft is not None and not ft:
                    ft = None

                try:
                    prompt = TranslatedPrompt(
                        prompt_uid=prompt_uid,
                        item_id=item_id,
                        facet=facet,
                        run=run,
                        lang=lang,
                        composed_prompt=raw["composed_prompt"],
                        composed_prompt_en=raw["composed_prompt_en"],
                        back_translation_en=raw["back_translation_en"],
                        comet_score=raw.get("comet_score"),
                        chrf_score=chrf_score,
                        word_overlap=raw.get("word_overlap"),
                        sanity_checks=raw.get("sanity_checks"),
                        fields_translated=ft,
                        domain_tag=raw.get("domain_tag", ""),
                        difficulty_tag=raw.get("difficulty_tag", ""),
                        severity_tag=raw.get("severity_tag", "none"),
                        delusion_type=raw.get("delusion_type", ""),
                    )
                    prompts.append(prompt)
                except Exception as e:
                    print(f"  ERROR: {path}:{line_num} — validation failed: {e}")
                    errors += 1

        print(f"  {lang}: {sum(1 for p in prompts if p.lang == lang)} prompts")

    if errors:
        print(f"\n  {errors} error(s) encountered during prompt adaptation")

    return prompts


def adapt_judges() -> list[JudgeTemplate]:
    """Read all per-language judge JSON files and convert to JSONL records."""
    templates: list[JudgeTemplate] = []
    errors = 0

    for lang in LANGUAGES:
        path = TANZIM_JUDGES_DIR / f"{lang}.json"
        if not path.exists():
            print(f"  WARNING: Missing {path}")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        facet_templates = data.get("templates", {})
        for facet, tdata in facet_templates.items():
            judge_uid = f"deepl__{lang}__{facet}"
            try:
                template = JudgeTemplate(
                    judge_uid=judge_uid,
                    facet=facet,
                    lang=lang,
                    judge_frame=tdata["judge_frame"],
                    judge_frame_en=tdata["back_translation_en"],
                    placeholders=tdata["placeholders"],
                    chain="deepl_deepl",
                )
                templates.append(template)
            except Exception as e:
                print(f"  ERROR: {path} facet={facet} — validation failed: {e}")
                errors += 1

        print(f"  {lang}: {sum(1 for t in templates if t.lang == lang)} judge templates")

    if errors:
        print(f"\n  {errors} error(s) encountered during judge adaptation")

    return templates


def write_output(
    prompts: list[TranslatedPrompt],
    templates: list[JudgeTemplate],
) -> None:
    """Write validated records to production JSONL files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts_path = OUTPUT_DIR / "prompts.jsonl"
    with open(prompts_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p.model_dump_json() + "\n")
    print(f"\n  Wrote {len(prompts)} prompts to {prompts_path}")

    judges_path = OUTPUT_DIR / "judges.jsonl"
    with open(judges_path, "w", encoding="utf-8") as f:
        for t in templates:
            f.write(t.model_dump_json() + "\n")
    print(f"  Wrote {len(templates)} judge templates to {judges_path}")


def main() -> None:
    print("=" * 60)
    print("TanzimSyco → SycoLingual Data Adapter")
    print("=" * 60)

    print(f"\nSource: {TANZIM_ROOT}")
    print(f"Target: {OUTPUT_DIR}")

    print("\n── Adapting prompts ──")
    prompts = adapt_prompts()

    print("\n── Adapting judge templates ──")
    templates = adapt_judges()

    print("\n── Writing output ──")
    write_output(prompts, templates)

    # Summary
    print("\n── Summary ──")
    print(f"  Total prompts:         {len(prompts)}")
    print(f"  Total judge templates: {len(templates)}")
    print(f"  Languages:             {len(LANGUAGES)}")

    # Sanity checks
    facets = {p.facet for p in prompts}
    langs = {p.lang for p in prompts}
    mirror_with_ft = sum(
        1 for p in prompts
        if p.facet == "mirror" and p.fields_translated
    )
    print(f"  Facets present:        {sorted(facets)}")
    print(f"  Languages present:     {sorted(langs)}")
    print(f"  Mirror items with fields_translated: {mirror_with_ft}")

    if len(prompts) != 2800:
        print(f"\n  WARNING: Expected 2800 prompts, got {len(prompts)}")
    if len(templates) != 40:
        print(f"\n  WARNING: Expected 40 judge templates, got {len(templates)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
