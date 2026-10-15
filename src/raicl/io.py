"""Saved response CSV schema and trial-identity validation."""
import csv
from pathlib import Path

from .retrieval import key_of


FIELDS = ["subject_id", "file_path", "true_label", "pred_label", "raw_response", "prompt_text", "few_shot_paths", "few_shot_distances", "error"]


def read_csv(path):
    csv.field_size_limit(10**9)
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    seen = set()
    for row in rows:
        key = key_of(row["file_path"])
        if key in seen or row["subject_id"] != key[0] or int(row["true_label"]) != key[1]:
            raise ValueError("Inconsistent or duplicate trial row")
        seen.add(key)
    return rows
