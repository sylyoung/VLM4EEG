"""Retrieval-augmented in-context learning for EEG waveform images.

The core API is CPU-only. Model runtimes are imported by their workflows,
not when importing this package.
"""
from .io import FIELDS, read_csv
from .metrics import DECISION, binary_metrics, parse_decision
from .prompts import prompt_from_paths
from .retrieval import KEY, evaluated, image_path, key_of, load_index, nearest, select_examples

__all__ = [
    "FIELDS", "KEY", "DECISION", "binary_metrics", "evaluated", "image_path",
    "key_of", "load_index", "nearest", "parse_decision", "prompt_from_paths",
    "read_csv", "select_examples",
]
