#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The three-class seizure-type prompt (non-seizure, focal, generalized), shared by the Gemini runner, the Qwen
evaluator and the fine-tuning set builder, so all of them send the same words (2026-10-12).

THE TEXT is the paper's, extended to three classes (prompts/, read once and logged with md5):
  base.txt                     prompt A (Base): task and output line only
  head.txt                     prompt Z's task, criteria and protocol, with a third criterion (seizure type by the
                               spatial extent of the activity) and a fourth protocol check (Extent)
  output_with_references.txt   prompt Z's output requirement
  output_no_references.txt     prompt B's (Reasoning) output requirement
ARMS: base (base.txt), reasoning (head, test, output without the comparison step), random and raicl (head, reference
block, test, output). The reference block is prompt Z's, "Reference Image i:", the image, "Label: <LABEL>.", with the
task classes first and the non-seizure anchors last, as prompt Z shows its seizure examples before its non-seizure
ones; its sentence states where the examples come from, which differs between random and raicl.

A prompt is text with one "[IMG: <path>]" marker per image (the form the Qwen evaluator and the paper's logs use);
gemini_content() turns it into the alternating text and image list the Gemini API takes. Decisions are parsed from the
last "FINAL DECISION:" line; anything else is undecided (-1).
"""
import ast
import hashlib
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = HERE
LABELS = ["NON-SEIZURE", "FOCAL SEIZURE", "GENERALIZED SEIZURE"]          # index = class id (y.npy)
ARMS = ["base", "reasoning", "random", "raicl"]
REFERENCE_ORDER = [1, 2, 0]           # task classes first, the non-seizure anchors last (prompt Z's order)
INTRO = "The following EEG-images serve as ground-truth anchors for the classification. "
PROVENANCE = {
    "raicl": "The NON-SEIZURE examples are from the same patient, whereas the FOCAL SEIZURE and GENERALIZED SEIZURE "
             "examples are from other patients with seizures:",
    "random": "The examples are from other patients:",
}
IMG_RE = re.compile(r"\[IMG: ([^\]]+)\]")
DECISION_RE = re.compile(r"FINAL\s+DECISION\s*:\s*\**\s*(NON-SEIZURE|FOCAL SEIZURE|GENERALIZED SEIZURE)")


def read_prompts():
    """{name: text}, and {name: md5}; trailing newlines stripped as eval_sleep_staging.py reads its files."""
    texts, md5 = {}, {}
    for name in ("base", "head", "output_with_references", "output_no_references"):
        with open(os.path.join(PROMPT_DIR, name + ".txt"), encoding="utf-8") as f:
            texts[name] = f.read().rstrip("\n")
        md5[name] = hashlib.md5(texts[name].encode("utf-8")).hexdigest()
    return texts, md5


def expected_labels(m):
    return [LABELS[k] for k in REFERENCE_ORDER for _ in range(m)]


def build_prompt(arm, test_path, ref_paths, ref_labels, texts, m=2):
    """The prompt text of one trial, with [IMG: path] markers. Raises ValueError on a malformed reference set."""
    if arm not in ARMS:
        raise ValueError("unknown arm %r" % arm)
    test = "[IMG: %s]" % test_path
    if arm in ("base", "reasoning"):
        if ref_paths or ref_labels:
            raise ValueError("%s: arm %s takes no references" % (test_path, arm))
        if arm == "base":
            if texts["base"].count("<Test EEG Image>") != 1:
                raise ValueError("base.txt must hold one <Test EEG Image> tag")
            return texts["base"].replace("<Test EEG Image>", test)
        return "\n\n".join([texts["head"], "[Test Input]\n" + test, texts["output_no_references"]])
    if list(ref_labels) != expected_labels(m) or len(ref_paths) != len(ref_labels):
        raise ValueError("%s: references %s, expected %s" % (test_path, ref_labels, expected_labels(m)))
    if test_path in ref_paths or len(set(ref_paths)) != len(ref_paths):
        raise ValueError("%s: a reference repeats or is the test image" % test_path)
    block = ["[Reference Examples]", INTRO + PROVENANCE[arm], ""]
    for i, (p, lab) in enumerate(zip(ref_paths, ref_labels), 1):
        block += ["Reference Image %d:" % i, "[IMG: %s]" % p, "Label: %s." % lab]
    return "\n\n".join([texts["head"], "\n".join(block), "[Test Input]\n" + test, texts["output_with_references"]])


def row_prompt(arm, row, texts, image_root, m=2):
    """build_prompt for a row of an input CSV (paths relative to image_root)."""
    refs = ast.literal_eval(row["few_shot_paths"]) if row.get("few_shot_paths") else []
    labels = ast.literal_eval(row["few_shot_labels"]) if row.get("few_shot_labels") else []
    return build_prompt(arm, os.path.join(image_root, row["file_path"]), [os.path.join(image_root, p) for p in refs],
                        labels, texts, m)


def n_images(arm, m=2):
    return 1 if arm in ("base", "reasoning") else 1 + len(REFERENCE_ORDER) * m


def parse_decision(raw):
    found = DECISION_RE.findall((raw or "").upper())
    return LABELS.index(found[-1]) if found else -1


def gemini_content(prompt, load_image):
    """[text, image, text, ...] in prompt order; load_image(path) returns the image object."""
    content, pos = [], 0
    for mt in IMG_RE.finditer(prompt):
        if mt.start() > pos:
            content.append(prompt[pos:mt.start()])
        content.append(load_image(mt.group(1)))
        pos = mt.end()
    if pos < len(prompt):
        content.append(prompt[pos:])
    return content
