# Results and model responses

[Repository](../README.md) · [Binary prompts](../prompts/README.md#binary-seizure-detection) · [Subtype prompts](../prompts/README.md#three-class-seizure-subtype)

## Main comparison tables

The tables compare electroencephalography (EEG) seizure detection on the [Wuhan Children's Hospital seizure dataset (CHSZ)](https://zenodo.org/records/19333249) and the [neonatal intensive care unit (NICU) dataset](https://zenodo.org/records/1280684) from Helsinki University Hospital, and seizure-subtype classification on the [Temple University Hospital EEG Seizure Corpus (TUSZ)](https://isip.piconepress.com/projects/tuh_eeg/).

| Task | Table image | Numerical table |
| --- | --- | --- |
| Binary seizure detection, CHSZ and NICU | [Full comparison](../figures/seizure_detection.png) | [seizure_detection.csv](seizure_detection.csv) |
| Three-class seizure subtype, TUSZ | [Full comparison](../figures/seizure_subtype.png) | [seizure_subtype.csv](seizure_subtype.csv) |

The tables report the current study results; the linked arXiv version is earlier. The CSVs transcribe the table values, not newly recomputed experiment results. Binary rows contain dataset, input type, model, four metric means and their standard deviations where reported. Empty standard-deviation cells mean no multi-seed deviation was reported. All values are percentages.

Signal models used five seeds and image models three. Baseline training queries were matched to the vision-language model (VLM) fine-tuning queries. The baselines do not receive the target patient's labeled non-seizure anchors, which are available to retrieval-augmented in-context learning (RAICL).

## CHSZ prediction workbooks

Metrics include balanced classification accuracy (BCA), sensitivity, specificity and macro F1. Qwen uses joint low-rank adaptation (LoRA).

| Model | BCA | Sensitivity | Specificity | Macro F1 | Complete responses |
| --- | ---: | ---: | ---: | ---: | --- |
| Gemini-3-Flash, untuned RAICL | 82.28 | **78.09** | 86.48 | 74.25 | [Workbook](CHSZ_Gemini-3-Flash_RAICL.xlsx) |
| Qwen3.5-9B, joint RAICL LoRA | **82.84** | 74.41 | **91.27** | **78.00** | [Workbook](CHSZ_Qwen3.5-9B_RAICL-finetuned.xlsx) |

Bold marks the higher value between these two rows. Each workbook has one sheet, covering the same 2,783 trials from 27 patients: all seizure trials and every tenth non-seizure trial. No patient images are included.

| Column | Meaning |
| --- | --- |
| `patient` | Patient identifier |
| `trial` | Global trial index in the preprocessed array |
| `true_label` | `SEIZURE` or `NON-SEIZURE` |
| `predicted_label` | Parsed final decision, or `UNANSWERED` |
| `reference_image_1_seizure` | First seizure reference in prompt order |
| `reference_image_2_seizure` | Second seizure reference in prompt order |
| `reference_image_3_non_seizure` | First same-patient background reference |
| `reference_image_4_non_seizure` | Second same-patient background reference |
| `response` | Complete model response |

Reference paths are dataset-relative. Their workbook order follows the prompt, not the retrieval function's internal return order. The existing Gemini workbook includes two stored-label corrections against the response's final decision; no responses or workbook bytes are changed by the documentation revision.

## Scoring

After installing the package, run from the repository root:

```sh
python scripts/score.py results/CHSZ_Gemini-3-Flash_RAICL.xlsx \
  results/CHSZ_Qwen3.5-9B_RAICL-finetuned.xlsx
```

The scorer checks the schema, unique trial identities, labels, agreement with response decisions, expected patient/trial counts and the non-seizure query rule. Tests additionally check the two trial lists and workbook hashes.

Binary metrics are computed separately for each patient, then averaged across patients:

- Sensitivity is seizure recall.
- Specificity is non-seizure recall.
- BCA is their mean.
- Macro F1 is the mean F1 of the two classes.

A response without a recognized final decision counts as non-seizure, not a dropped row. Neither distributed workbook contains an unanswered response. A network failure, generation failure or empty response is not a prediction.

For TUSZ, BCA averages recall over the ground-truth classes present in each patient's query list. Specificity and macro F1 average their one-versus-rest values over those same classes. The final metrics then average across patients. The public binary workbook scorer does not implement this three-class evaluation.

## Comparison models

The paper's comparison models are listed below; their implementations and per-trial outputs are not included in this repository. The models include convolutional neural networks (CNNs) and Vision Transformers (ViTs).

| Models | Description | Reference |
| --- | --- | --- |
| ShallowCNN, DeepCNN | Temporal/spatial convolutional models operating on raw EEG | Human Brain Mapping, 2017 |
| EEGNet | Compact depthwise and separable convolutional EEG network | J. Neural Eng., 2018 |
| Conformer | CNN with Transformer blocks for temporal dependencies | IEEE TNSRE, 2023 |
| Deformer | Dense convolutional Transformer for temporal dynamics | IEEE JBHI, 2025 |
| ADFCNN | Dual-scale convolution with attention-based feature fusion | IEEE TNSRE, 2024 |
| DBConformer | Dual branches for temporal and spatial representation | IEEE JBHI, 2026 |
| GoogLeNet, ResNet, DenseNet | Multi-scale, residual and densely connected image classifiers | CVPR 2015, 2016, 2017 |
| EfficientNet, Swin Transformer, ViT | Compound-scaled CNN and Transformer image classifiers | ICML 2019, ICCV 2021, ICLR 2021 |

## Available outputs

Only the two CHSZ response workbooks are distributed. The full comparison CSVs also report NICU, TUSZ and baseline summary results, but are not substitutes for those experiments' per-trial predictions. Untuned Qwen, other backbones and ablation response sets are not included.

Generated experiments belong in the ignored `outputs/` directory. The four-second trial metrics do not measure event-level sensitivity, false alarms per hour or clinical readiness.
