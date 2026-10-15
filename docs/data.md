# Data and waveform images

[Back to the repository overview](../README.md)

The pipeline starts from **preprocessed NumPy arrays** of electroencephalography (EEG) signals, not raw European Data Format (EDF) recordings. This repository does not redistribute raw EEG, patient images, or a raw-recording preprocessing pipeline.

## Download CHSZ

The **[EEG-Based Epileptic Seizure Detection Dataset (CHSZ)](https://zenodo.org/records/19333249)** contains pediatric recordings from Wuhan Children's Hospital. The Zenodo record provides 27 per-patient MATLAB files, `prepro_subject_1.mat` through `prepro_subject_27.mat`, under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/). Follow the record's citation and usage requirements.

The download is not the ready-to-use NumPy input below. Its trial arrays use `(trials, samples, channels)` with 18 channels and 2,000 or 4,000 samples per four-second trial, corresponding to 500 or 1,000 Hz. For this repository's experiments, resample each trial to **250 Hz**, arrange it as `(trials, channels, samples)`, and concatenate patients in the original study order. The record's suggested 500 Hz rate is not the rate used here. Preserve labels and within-patient trial order, and verify each patient's trial count against [`render.py`](../scripts/render.py) before combining the arrays.

## Expected arrays

Place your arrays under an ignored local `data/` directory:

```text
data/CHSZ/
├── X.npy                 # (21227, 18, 1000): trials × channels × samples
└── y.npy                 # (21227,): 0 = non-seizure, 1 = seizure
```

CHSZ uses non-overlapping four-second trials at 250 Hz. The manuscript's preprocessing uses an 18-channel bipolar montage, a 50 Hz notch filter, and a 0.5–50 Hz band-pass filter before segmentation, with resampling to 250 Hz.

**Patient ordering is essential.** The renderer maps contiguous trial blocks to the 27 patients using the original count list in [`render.py`](../scripts/render.py). Its shape and total-count checks cannot prove that a supplied array uses the correct patient order. Do not substitute another ordering or guess identities from array dimensions.

The bipolar channels are ordered as follows:

```text
Fp2-F4, F4-C4, C4-P4, P4-O2,
Fp1-F3, F3-C3, C3-P3, P3-O1,
Fp2-F8, F8-T4, T4-T6, T6-O2,
Fp1-F7, F7-T3, T3-T5, T5-O1,
Fz-Cz, Cz-Pz
```

## Render paired images

Install the rendering dependencies and run:

```bash
python -m pip install -e ".[render]"
raicl-render --data_dir ./data --dataset CHSZ --workers 4
```

The output layout is:

```text
data/CHSZ/time-chan/
├── Vision/
│   └── sub_XX/class_Y/trial_N_Vision.png
├── VLM/
│   └── sub_XX/class_Y/trial_N_VLM.png
└── simulation_checks/    # trial-0 model-input illustrations
```

`N` is the **global trial index** in the input arrays, not a within-patient counter. Patient IDs such as `sub_00` are release identifiers. `class_0` and `class_1` mean non-seizure and seizure.

The `Vision` and `VLM` files for a trial must share patient, class and global trial identity. Contrastive Language-Image Pre-training (CLIP) reads `Vision`; the vision-language model (VLM) reads `VLM`.

`Vision` images omit axes and labels and use a lane-clipping fraction of 0.95. `VLM` images retain channel names and the time axis and use 0.85. Each channel is scaled by its own 1st and 99th percentiles; this does not preserve an absolute microvolt scale.

## Extract CLIP features

Install hardware-appropriate PyTorch/torchvision and OpenAI CLIP:

```sh
python -m pip install -e ".[embed]"
python -m pip install git+https://github.com/openai/CLIP.git
mkdir -p outputs
python scripts/embed.py --data_dir ./data --dataset CHSZ \
  --image_type Vision --index_from Vision \
  --index_out outputs/index_CHSZ_RawVectors.pkl
```

Features are normalized 512-dimensional vectors from CLIP's base Vision Transformer with 32-pixel patches (ViT-B/32), stored as float16 under `Vision_Embeddings/`. The index records each vector's image path, patient, class, and trial ID. **Load only an index you generated or otherwise trust**: pickle can execute code when opened.

The index preserves image paths. Relative paths are resolved from your working directory; absolute paths must remain accessible. Run the documented data workflows from the repository root. Moving the data directory requires rebuilding the index and query lists, not editing a live result file.

## Evaluation and training lists

On CHSZ the fixed evaluation rule includes every seizure trial and every tenth non-seizure trial, identified by `N % 10 == 0`. The released workbooks contain 2,783 evaluation trials across 27 patients.

The [packaged training-query manifest](../configs/chsz_loso_training_queries.json) retains the original ordered identifiers for all 27 leave-one-patient-out folds. Each fold has all auxiliary-patient seizure queries and the same number of non-seizure queries outside the evaluation list. The held-out patient supplies neither training queries nor fine-tuning reference images.

## NICU and TUSZ

The renderer and binary Gemini runner also support the **[neonatal intensive care unit (NICU) dataset](https://zenodo.org/records/1280684)** from Helsinki University Hospital. NICU uses 39 retained patients, arrays shaped `(54073, 18, 1024)`, four-second trials at 256 Hz, and its own fixed patient ordering. Its evaluation rule selects every tenth trial of both classes. **NICU prediction workbooks and fine-tuning query manifests are not released here.** The public fold-preparation command is CHSZ-specific.

The **[Temple University Hospital EEG Seizure Corpus (TUSZ)](https://isip.piconepress.com/projects/tuh_eeg/)** supplies the three-class task with labels 0 = non-seizure, 1 = focal seizure and 2 = generalized seizure. The retained evaluation subset contains 29 patients and 80,530 four-second trials before query selection. Preprocessing uses 18 bipolar channels, a 60 Hz notch filter, a 0.5–50 Hz band-pass filter and resampling to 250 Hz. The fixed evaluation list has 1,721 queries. The [complete prompt variants](../prompts/README.md#three-class-seizure-subtype) and [comparison table](../results/seizure_subtype.csv) are provided, but the binary runners do not execute this task. No TUSZ prediction workbook or raw-recording preprocessing pipeline is included.
