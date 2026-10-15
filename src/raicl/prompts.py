"""Assemble labeled images in the original benchmark prompt order."""
import re
from pathlib import Path

from .paths import DEFAULT_PROMPT
from .retrieval import image_path


def prompt_from_paths(test_path, reference_paths, template=DEFAULT_PROMPT):
    paths = list(reference_paths)
    if len(paths) not in (0, 4):
        raise ValueError("The released prompt requires M=2 references per class")
    text = Path(template).read_text(encoding="utf-8").rstrip("\n")
    mapping = {"<Test EEG Image>": image_path(test_path)}
    if paths:
        for tag, path in zip(["<Nearest Neighbor Example 1>", "<Nearest Neighbor Example 2>", "<Non-Seizure Example EEG Image 1>", "<Non-Seizure Example EEG Image 2>"], [paths[2], paths[3], paths[0], paths[1]]):
            mapping[tag] = image_path(path)
    pattern = re.compile("|".join(re.escape(tag) for tag in mapping))
    used = []
    def substitute(match):
        path = mapping[match.group()]
        used.append(path)
        return "[IMG: " + path + "]"
    text = pattern.sub(substitute, text)
    if re.search(r"<(?:Nearest Neighbor Example|Non-Seizure Example EEG Image|Test EEG Image)", text):
        raise ValueError("Prompt contains an unresolved image slot")
    if len(used) != (5 if paths else 1):
        raise ValueError("Prompt does not contain exactly the expected images")
    return text, used
