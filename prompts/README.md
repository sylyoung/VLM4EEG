# EEG prompts

[Repository](../README.md) · [Gemini API](../docs/gemini.md) · [Fine-tuning](../docs/finetuning.md)

These are the exact prompt words for electroencephalography (EEG) classification, with image placeholders left visible. Retrieval-augmented in-context learning (RAICL) supplies representative and similar labeled examples. Binary templates and the original Temple University Hospital EEG Seizure Corpus (TUSZ) components are unchanged. The full TUSZ Reasoning, Random and RAICL files are expanded from the original builder so that each configuration can be read in one file.

## Binary seizure detection

Labels are `NON-SEIZURE` and `SEIZURE`.

| Configuration | Full prompt | Images | Content |
| --- | --- | ---: | --- |
| Base | [binary/base.txt](binary/base.txt) | 1 | Task and decision only |
| Reasoning | [binary/reasoning.txt](binary/reasoning.txt) | 1 | Expert criteria and analysis before the decision |
| Random | [binary/random_examples.txt](binary/random_examples.txt) | 5 | Expert criteria and four randomly selected examples |
| RAICL | [binary/raicl.txt](binary/raicl.txt) | 5 | Expert criteria and four RAICL-selected examples |

The RAICL template was previously named `prompt-Z.txt`. Its image slots appear in this order:

1. `<Nearest Neighbor Example 2>` — seizure.
2. `<Nearest Neighbor Example 1>` — seizure.
3. `<Non-Seizure Example EEG Image 1>` — non-seizure anchor.
4. `<Non-Seizure Example EEG Image 2>` — non-seizure anchor.
5. `<Test EEG Image>` — query.

The reversal of the two seizure slots is part of the original template. The core assembler preserves it:

```python
from raicl import prompt_from_paths

# The input list has non-seizure anchors first, then seizure references.
references = ["background_1_VLM.png", "background_2_VLM.png",
              "seizure_1_VLM.png", "seizure_2_VLM.png"]
text, images = prompt_from_paths("query_VLM.png", references)
```

`text` contains `[IMG: path]` markers. `images` follows their order in the prompt. For Base or Reasoning, call `prompt_from_paths` with an empty reference list and the corresponding template path.

The archived Random template uses `<Seizure Example EEG Image 1>` and `<Seizure Example EEG Image 2>` instead of the nearest-neighbor tags. Its provenance sentence still says the non-seizure examples are from the same patient, although the paper's Random condition samples auxiliary-patient examples. The original wording is retained here rather than silently corrected. This template is not interchangeable with the binary RAICL runner's `--prompt` argument.

## Three-class seizure subtype

Labels are `NON-SEIZURE`, `FOCAL SEIZURE` and `GENERALIZED SEIZURE`.

| Configuration | Full prompt | Images | Content |
| --- | --- | ---: | --- |
| Base | [tusz/base.txt](tusz/base.txt) | 1 | Three-class task and decision |
| Reasoning | [tusz/reasoning.txt](tusz/reasoning.txt) | 1 | Expert criteria, including spatial extent |
| Random | [tusz/random.txt](tusz/random.txt) | 7 | Expert criteria and two auxiliary-patient examples per class |
| RAICL | [tusz/raicl.txt](tusz/raicl.txt) | 7 | Expert criteria, focal/generalized references and patient-specific background anchors |

The full example-based templates have two focal references, two generalized references, two non-seizure references, then the test image. Angle-bracket image slots are substituted without changing the surrounding text. Generate a prompt with real image paths using the original [builder](tusz/tusz_seizure_type_prompt.py):

```python
from raicl.resources.prompts.tusz.tusz_seizure_type_prompt import (
    build_prompt, expected_labels, read_prompts,
)

texts, checksums = read_prompts()
references = ["focal_1.png", "focal_2.png",
              "generalized_1.png", "generalized_2.png",
              "background_1.png", "background_2.png"]
prompt = build_prompt("raicl", "query.png", references,
                      expected_labels(2), texts, m=2)
```

For `random`, supply the randomly selected references in the same class order. For `base` or `reasoning`, supply empty reference and label lists. The builder validates labels and prevents duplicate references or reuse of the query.

The shared components remain available as [head.txt](tusz/head.txt), [output_with_references.txt](tusz/output_with_references.txt) and [output_no_references.txt](tusz/output_no_references.txt). The builder assembles words and images; it does not select TUSZ references or run a dataset evaluation. The public dataset runners are binary-only.

## Reference-selection variants

These are selection ablations, not additional wording templates:

| Strategy | Non-seizure examples | Seizure examples |
| --- | --- | --- |
| Random | Random auxiliary-patient trials | Random auxiliary-patient trials |
| Resting-State Anchor | Random earlier trials from the test patient | Random auxiliary-patient trials |
| Representativeness | Test-patient trials nearest to the background centroid | Representative auxiliary-patient trials |
| RAICL: Representativeness + Similarity | Test-patient trials nearest to the background centroid | Auxiliary-patient seizure medoids nearest to the query |

For multiple seizure classes, apply the seizure selection independently to each class. The public binary retrieval function implements the final RAICL rule; the table describes the paper's ablations, not four command-line modes.

## Additional wording controls

- [experimental/concrete_wording.txt](experimental/concrete_wording.txt) is a concrete-wording control.
- [experimental/decision_only.txt](experimental/decision_only.txt) asks for a decision without a rationale and avoids numerical frequency and absolute-amplitude claims.

Neither template produced the two downloadable workbook results. Qwen fine-tuning uses decision-line targets but retains the original RAICL input prompt; decision-only supervision and a decision-only input template are different choices.

## Output and scoring

Binary responses end in `FINAL DECISION: SEIZURE` or `FINAL DECISION: NON-SEIZURE`. Three-class responses end in `FINAL DECISION: NON-SEIZURE`, `FINAL DECISION: FOCAL SEIZURE` or `FINAL DECISION: GENERALIZED SEIZURE`.

The parser takes the last recognized final decision. A response without one counts as non-seizure during seizure-task scoring; a failed request is not an unanswered response. Per-channel waveform scaling and image resolution limit the interpretation of the prompts' amplitude and frequency language. Keep a separate result/settings file whenever wording, reference selection or output requirements change.
