"""CLI entry point for SycoLingual v2 evaluation pipeline."""

import argparse
import asyncio
import sys

from src.config import load_config


def cmd_evaluate(args):
    from src.runner import EvaluationRunner
    config = load_config(args.config)
    runner = EvaluationRunner(
        config, dry_run=args.dry_run, model_filter=args.model,
    )
    asyncio.run(runner.run())


def cmd_judge(args):
    from src.judge import JudgingModule
    config = load_config(args.config)
    module = JudgingModule(config, dry_run=args.dry_run)
    if args.aggregate:
        module.aggregate(config.paths.judgements, config.paths.judgements.replace(".jsonl", "_scored.jsonl"))
    else:
        asyncio.run(module.run(english_validation_only=args.english_validation))


def cmd_status(args):
    from src.config import load_config
    from src.io import load_completed_keys
    config = load_config(args.config)
    eval_keys = load_completed_keys(config.paths.responses, ["prompt_id", "model"])
    judge_keys = load_completed_keys(config.paths.judgements, ["prompt_id", "model", "judge_model"])
    print(f"Evaluation: {len(eval_keys)} responses completed")
    print(f"Judging: {len(judge_keys)} judge scores completed")


def main():
    parser = argparse.ArgumentParser(description="SycoLingual v2 Evaluation Pipeline")
    parser.add_argument("--config", default="config/experiment.yaml", help="Config file path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run model evaluations (step D)")
    eval_parser.add_argument("--dry-run", action="store_true", help="Use mock providers")
    eval_parser.add_argument("--model", type=str, help="Run only this model")
    eval_parser.set_defaults(func=cmd_evaluate)

    # judge
    judge_parser = subparsers.add_parser("judge", help="Run judging panel (step F)")
    judge_parser.add_argument("--dry-run", action="store_true", help="Use mock providers")
    judge_parser.add_argument("--english-validation", action="store_true", help="Run English validation subset only")
    judge_parser.add_argument("--aggregate", action="store_true", help="Recompute medians from existing scores")
    judge_parser.set_defaults(func=cmd_judge)

    # status
    status_parser = subparsers.add_parser("status", help="Show pipeline progress")
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
