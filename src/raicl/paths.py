"""Benchmark resources available from both the source tree and an installed package."""
from pathlib import Path

from .resources import configs, prompts

PROMPTS = Path(prompts.__file__).resolve().parent
DEFAULT_PROMPT = PROMPTS / "binary/raicl.txt"
TRAINING_MANIFEST = Path(configs.__file__).resolve().parent / "chsz_loso_training_queries.json"
