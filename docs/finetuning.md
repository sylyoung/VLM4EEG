# Fine-tuning Qwen3.5-9B

[Repository](../README.md) · [Data preparation](data.md) · [Exact prompts](../prompts/README.md)

This is the leave-one-patient-out training path for the [Wuhan Children's Hospital seizure dataset (CHSZ)](https://zenodo.org/records/19333249). A separate low-rank adaptation (LoRA) adapter is trained for each held-out patient, using class-balanced queries from the other patients. Every query includes retrieval-augmented in-context learning (RAICL) references; the target contains only its final decision.

## Installation

Use Linux with an NVIDIA CUDA-compatible graphics processing unit (GPU). The core package supports Python 3.10+, but current LLaMA-Factory requires Python 3.11+. A separate environment avoids changing an existing central processing unit (CPU) setup:

```sh
conda create -n raicl-qwen python=3.11 pip
conda activate raicl-qwen
python -m pip install -e ".[render,embed]"
python -m pip install git+https://github.com/openai/CLIP.git
```

Install a hardware-compatible PyTorch/torchvision build using [PyTorch's CUDA instructions](https://pytorch.org/get-started/locally/). Then install LLaMA-Factory, bitsandbytes and the model-download client:

```sh
git clone https://github.com/hiyouga/LLaMA-Factory.git external/LLaMA-Factory
python -m pip install -e ./external/LLaMA-Factory
python -m pip install bitsandbytes huggingface_hub

hf download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B
```

Use a LLaMA-Factory/Transformers combination that supports both **Qwen3.5** and the **`qwen3_5` template**. Check before preparing a long run:

```sh
python -c "from llamafactory.data.template import TEMPLATES; assert 'qwen3_5' in TEMPLATES"
python -c "from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'"
```

These commands check availability, not an end-to-end training run. Dependencies here are installation requirements, not a frozen reproduction environment. Record the LLaMA-Factory commit, package versions, CUDA version and model revision with your experiment. The supplied helpers use LLaMA-Factory's Hugging Face backend and internal model interfaces; upstream changes can require adaptation.

## Prepare images and retrieval features

Follow [data preparation](data.md) to render electroencephalography (EEG) waveform images and extract Contrastive Language-Image Pre-training (CLIP) features. From the repository root:

```sh
python scripts/render.py --data_dir ./data --dataset CHSZ --workers 4
mkdir -p outputs
python scripts/embed.py --data_dir ./data --dataset CHSZ \
  --image_type Vision --index_from Vision \
  --index_out outputs/index_CHSZ_RawVectors.pkl
```

The index must match the exact patient/trial identities used by [`chsz_loso_training_queries.json`](../configs/chsz_loso_training_queries.json). Never load an untrusted pickle index.

## Prepare one fold

```sh
python scripts/prepare.py --index outputs/index_CHSZ_RawVectors.pkl \
  --subject sub_00 --out-dir outputs/folds/sub_00 \
  --training --model ./models/Qwen3.5-9B
```

This produces:

```text
outputs/folds/sub_00/
  sub_00_raicl.csv                  # held-out patient's test queries and references
  chsz_raicl_loso_sub_00.jsonl       # training messages, decision targets and image paths
  dataset_info.json                # LLaMA-Factory ShareGPT dataset registration
  chsz_raicl_loso_sub_00.yaml        # complete training configuration
  protocol.json                    # query counts, held-out patient and selection rule
```

The builder preserves the manifest's query order and checks class balance. The held-out patient is excluded from both training queries and training references. Non-seizure training queries lie outside the CHSZ evaluation list. Omitting `--training` creates only the inference CSV.

Each JSON Lines (JSONL) sample contains a user message with `<image>` markers, an assistant message such as `FINAL DECISION: SEIZURE`, and an `images` list in prompt order. There is no supervised rationale.

## Train

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  outputs/folds/sub_00/chsz_raicl_loso_sub_00.yaml
```

The generated default configuration uses:

| Parameter | Value |
| --- | --- |
| Modules adapted by LoRA | Visual encoder, projector and language model jointly |
| Rank / alpha / dropout | 32 / 16 / 0.05 |
| Base weights | 4-bit NormalFloat (NF4), double quantization, packed uint8 storage |
| Learning rate / epochs | `1e-4` / 2 |
| Batch per device / gradient accumulation | 1 / 16 |
| Schedule / warmup ratio | Cosine / 0.05 |
| Cutoff length | 4,096 tokens |
| Precision / gradient checkpointing | Brain floating point, 16-bit (bf16) / enabled |
| Seed | 42 |
| Target | Final decision line only |
| Checkpoint selection | Final adapter; no selection using held-out-patient scores |

The [decision-only loss helper](../src/raicl/models/answer_only_logits.py) computes logits at supervised token positions and requires a per-device batch of one. The normal generation forward method remains in use at inference. The training wrapper checks the batch/accumulation assumptions and refuses a nonempty output directory unless resume is explicit.

To resume a known interrupted checkpoint:

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  outputs/folds/sub_00/chsz_raicl_loso_sub_00.yaml \
  --resume-from-checkpoint outputs/folds/sub_00/adapter/checkpoint-N
```

Replace `checkpoint-N` with an existing checkpoint directory. Do not overwrite an unrelated adapter.

## Evaluate the final adapter

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/qwen.py \
  --model ./models/Qwen3.5-9B \
  --adapter outputs/folds/sub_00/adapter \
  --csv outputs/folds/sub_00/sub_00_raicl.csv \
  --out outputs/CHSZ_qwen_sub_00_raicl.csv \
  --batch-size 4 --max-new-tokens 8192
```

Inference uses greedy decoding, 4-bit weights, batch size 4 and an 8,192-token generation budget. The output retains complete responses, references and an adjacent settings record. `--resume` requires identical settings. Empty responses and generation failures stop the run; an actual answer without a recognized decision is scored as non-seizure.

Repeat preparation, training and evaluation for `sub_00` through `sub_26`. Use each patient's own adapter and query list. Compute metrics within each patient, then average across all patients; a single fold is not the full CHSZ result.

## Training variants and scope

The paper compares joint adaptation with visual-encoder/projector-only and language-model-only adaptation, and with training without examples or with random examples. The supplied fold builder generates the default **joint RAICL-example** configuration; it does not generate every ablation training set. Changing freeze flags is not equivalent to reproducing the no-example or random-example training conditions.

The original inference prompt asks for analysis even though training targets contain only the final decision. This distinction is preserved for reproduction. Substituting the experimental decision-only prompt defines another condition.

Training manifests for the neonatal intensive care unit (NICU) dataset and the Temple University Hospital EEG Seizure Corpus (TUSZ), a complete three-class pipeline, other-backbone trainers and pretrained adapters are not distributed. The [TUSZ prompts](../prompts/README.md#three-class-seizure-subtype) are available independently of this CHSZ training path.
