"""Send one waveform PNG to Gemini with the binary or three-class Base prompt."""
import argparse
import os
from pathlib import Path

from raicl.paths import PROMPTS


def request_parts(text, image_path, types):
    before, after = text.split("<Test EEG Image>")
    return [types.Part.from_text(text=before),
            types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png"),
            types.Part.from_text(text=after)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Existing waveform image in PNG format")
    parser.add_argument("--task", choices=["binary", "tusz"], default="binary")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    args = parser.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        parser.error("Set GEMINI_API_KEY in your environment; do not put it in this file")
    if not args.image.is_file():
        parser.error("The waveform image does not exist")
    if not args.image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        parser.error("The input must be a PNG waveform image")
    from google import genai
    from google.genai import types
    text = (PROMPTS / args.task / "base.txt").read_text().rstrip("\n")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=180000))
    try:
        response = client.models.generate_content(
            model=args.model,
            contents=request_parts(text, args.image, types),
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=2048),
        )
        if not response.text:
            raise RuntimeError("Gemini returned no response text; this is not a prediction")
        print(response.text)
    finally:
        client.close()


if __name__ == "__main__":
    main()
