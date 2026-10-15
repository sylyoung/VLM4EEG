# Examples

The Gemini example sends electroencephalography (EEG) waveform images to the Gemini application programming interface (API).

| Example | Command | Requirements |
| --- | --- | --- |
| [One image through Gemini](gemini_api.py) | `python examples/gemini_api.py --image path/to/waveform.png` | `.[gemini]`, API key and a PNG; makes one API request |
| [Synthetic retrieval](retrieval_demo.py) | `python examples/retrieval_demo.py` | Core package only; no EEG images, API calls or model weights |

Run from the repository root after installation. The Gemini example uses the exact Base prompt; add `--task tusz` for three classes. For automatic reference retrieval and full binary inference, use [the retrieval-augmented in-context learning (RAICL) runner](../README.md#gemini-api).
