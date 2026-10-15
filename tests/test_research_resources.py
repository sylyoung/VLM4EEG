"""Check discoverable scripts, complete prompts and the Gemini example without model calls."""
import csv
import importlib.util
from pathlib import Path
import re
import subprocess
import sys

import pytest

from raicl.paths import PROMPTS
from raicl.resources.prompts.tusz import tusz_seizure_type_prompt as tusz

ROOT = Path(__file__).resolve().parents[1]


def load_example():
    spec = importlib.util.spec_from_file_location("gemini_api_example", ROOT / "examples/gemini_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", ["render", "embed", "prepare", "gemini", "train", "qwen", "score"])
def test_visible_script_help_is_offline(script, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / (script + ".py")), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_gemini_example_help_does_not_need_a_key(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "examples/gemini_api.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert "--task" in result.stdout


@pytest.mark.parametrize("arm", ["reasoning", "random", "raicl"])
def test_full_subtype_templates_match_original_builder(arm):
    texts, _ = tusz.read_prompts()
    refs = [f"<{label} Example EEG Image {i}>"
            for label in ("Focal Seizure", "Generalized Seizure", "Non-Seizure") for i in (1, 2)]
    full = tusz.build_prompt(arm, "<Test EEG Image>", refs if arm != "reasoning" else [],
                             tusz.expected_labels(2) if arm != "reasoning" else [], texts, m=2)
    full = re.sub(r"\[IMG: (<[^>]+>)\]", r"\1", full)
    assert (PROMPTS / "tusz" / (arm + ".txt")).read_text().rstrip("\n") == full
    assert full.count("<Test EEG Image>") == 1
    assert len(re.findall(r"<[^>]+>", full)) == (1 if arm == "reasoning" else 7)


@pytest.mark.parametrize("task", ["binary", "tusz"])
def test_gemini_example_sends_exact_base_text_and_inline_image(task, tmp_path):
    class Part:
        @staticmethod
        def from_text(*, text):
            return ("text", text)

        @staticmethod
        def from_bytes(*, data, mime_type):
            return (mime_type, data)

    class Types:
        Part = None

    Types.Part = Part
    image = tmp_path / "test.png"
    image.write_bytes(b"synthetic image bytes; no model request")
    text = (PROMPTS / task / "base.txt").read_text().rstrip("\n")
    parts = load_example().request_parts(text, image, Types)
    assert len(parts) == 3
    assert parts[1] == ("image/png", image.read_bytes())
    assert parts[0][1] + "<Test EEG Image>" + parts[2][1] == text


def test_full_binary_table_contains_the_released_workbook_metrics():
    with (ROOT / "results/seizure_detection.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 30
    assert {row["dataset"] for row in rows} == {"CHSZ", "NICU"}
    assert len({row["model"] for row in rows}) == 15
    expected = {
        "Gemini-3-Flash": ["82.28", "78.09", "86.48", "74.25"],
        "Qwen3.5-9B, fine-tuned": ["82.84", "74.41", "91.27", "78.00"],
    }
    for row in rows:
        if row["dataset"] == "CHSZ" and row["model"] in expected:
            assert [row[key] for key in ("BCA", "Sensitivity", "Specificity", "Macro F1")] == expected[row["model"]]


def test_subtype_table_keeps_models_and_prompt_variants_separate():
    with (ROOT / "results/seizure_subtype.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 9
    assert {row["prompt"] for row in rows} == {"Base", "Random", "RAICL"}
    assert len({row["model"] for row in rows}) == 3
    assert len({(row["model"], row["prompt"]) for row in rows}) == 9
