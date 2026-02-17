"""Tests for the CLI entry point (run.py)."""

import subprocess
import sys


class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "run.py", "--help"],
            capture_output=True, text=True,
            cwd="/home/alexc/projects/syco-lingual",
        )
        assert result.returncode == 0
        assert "evaluate" in result.stdout
        assert "judge" in result.stdout
        assert "status" in result.stdout

    def test_evaluate_help(self):
        result = subprocess.run(
            [sys.executable, "run.py", "evaluate", "--help"],
            capture_output=True, text=True,
            cwd="/home/alexc/projects/syco-lingual",
        )
        assert result.returncode == 0
        assert "--dry-run" in result.stdout
        assert "--model" in result.stdout

    def test_judge_help(self):
        result = subprocess.run(
            [sys.executable, "run.py", "judge", "--help"],
            capture_output=True, text=True,
            cwd="/home/alexc/projects/syco-lingual",
        )
        assert result.returncode == 0
        assert "--english-validation" in result.stdout
        assert "--aggregate" in result.stdout
