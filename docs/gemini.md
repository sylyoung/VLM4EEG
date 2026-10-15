# Gemini API

[Repository](../README.md) · [Exact prompts](../prompts/README.md) · [Data preparation](data.md)

Use the Gemini application programming interface (API) to classify electroencephalography (EEG) waveform images. Authentication and requests use Google's Gen AI software development kit (SDK). Retrieval-augmented in-context learning (RAICL) adds selected labeled examples to the query.

## Authentication

1. Create a key in [Google AI Studio](https://aistudio.google.com/apikey). Google's [API-key guide](https://ai.google.dev/gemini-api/docs/api-key) explains projects, permissions and key restrictions.
2. Install the SDK from the repository root:

   ```sh
   python -m pip install -e ".[gemini]"
   ```

3. Enter the key without placing it in shell history:

   ```sh
   read -rs GEMINI_API_KEY
   export GEMINI_API_KEY
   ```

   Paste the key while `read` waits, then press Enter. Input is hidden. The repository reads `GEMINI_API_KEY` explicitly and never writes its value to outputs. Do not put a real key in Python files, notebooks, prompts or commits. Unset it with `unset GEMINI_API_KEY` when finished.

Requests leave your machine. Check the dataset's permissions and Google's current API data-handling and billing terms before sending EEG images. The examples do not anonymize images automatically.

## One image through the SDK

The [minimal example](../examples/gemini_api.py) takes a PNG waveform image and the exact Base prompt. It preserves the template's text/image order, sends one request, prints the full response and closes its client:

```sh
python examples/gemini_api.py --image path/to/your_waveform.png
python examples/gemini_api.py --task tusz --image path/to/your_waveform.png
```

Replace the image path with a real file. The first command uses binary seizure detection; the second uses the three-class Base prompt for the Temple University Hospital EEG Seizure Corpus (TUSZ). Each command makes a request and may incur charges.

The underlying API calls are:

```python
import os
from pathlib import Path
from google import genai
from google.genai import types
from raicl.paths import PROMPTS

image = Path("path/to/your_waveform.png")
text = (PROMPTS / "binary/base.txt").read_text().rstrip("\n")
before, after = text.split("<Test EEG Image>")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
try:
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            types.Part.from_text(text=before),
            types.Part.from_bytes(data=image.read_bytes(), mime_type="image/png"),
            types.Part.from_text(text=after),
        ],
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=2048),
    )
    if not response.text:
        raise RuntimeError("No response text; this is not a prediction")
    print(response.text)
finally:
    client.close()
```

`gemini-3-flash-preview` is the recorded model ID, not a recommendation to substitute a newer model. API availability changes. If this ID is unavailable to your project, stop or explicitly choose a new model and report it as a different experiment; do not reuse the historical score.

## Binary RAICL over a dataset

The binary runner supports the [Wuhan Children's Hospital seizure dataset (CHSZ)](https://zenodo.org/records/19333249) and the [neonatal intensive care unit (NICU) dataset](https://zenodo.org/records/1280684) from Helsinki University Hospital. Prepare the [waveform images and Contrastive Language-Image Pre-training (CLIP) index](data.md), then run from the repository root:

```sh
python scripts/gemini.py --dataset CHSZ --data_dir ./data \
  --index outputs/index_CHSZ_RawVectors.pkl \
  --prompt prompts/binary/raicl.txt \
  --model gemini-3-flash-preview \
  --out outputs/CHSZ_gemini_raicl.csv
```

For NICU, change the dataset and index together. The runner uses two non-seizure anchors from the test patient and two seizure medoids from auxiliary patients. It does not execute Base, Reasoning, Random or TUSZ dataset evaluations simply by changing `--prompt`.

Use `--limit 1` with a **new output filename** for a one-trial RAICL check before starting the full evaluation. Do not use that limited run's output as a complete result. For a full run, use another output filename without `--limit`.

## Outputs and resume

The response CSV records patient/trial identity, true and predicted labels, complete response text, the assembled prompt, reference paths and distances, and parse status. Its neighboring `.settings.json` records model, temperature, token budget, evaluation rule and SHA-256 hashes of the prompt and retrieval index.

- Existing outputs are not overwritten.
- `--resume` requires the same saved settings, prompt and index.
- Request failures and empty responses stop the run and are not written as predictions.
- A completed response without a recognized final decision is saved as a parse failure and scored as non-seizure.
- CHSZ evaluates every seizure and every tenth non-seizure trial. NICU evaluates every tenth trial of both classes.

Score a saved binary CSV without making API calls:

```sh
python scripts/gemini.py --dataset CHSZ \
  --test_file outputs/CHSZ_gemini_raicl.csv
```

This scores the rows present; it does not certify that a partial CSV covers the complete evaluation list. For the two distributed workbooks, use [`scripts/score.py`](../scripts/score.py), which checks the expected trial and patient counts.
