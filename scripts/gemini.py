#!/usr/bin/env python3
"""Query Gemini with the released RAICL prompt, or score saved responses without API calls."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from raicl import FIELDS, binary_metrics, evaluated, image_path, key_of, load_index, parse_decision, prompt_from_paths, read_csv, select_examples
from raicl.paths import DEFAULT_PROMPT


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["CHSZ", "NICU"], required=True)
    parser.add_argument("--data_dir", type=Path, help="Root containing CHSZ/time-chan/{Vision,VLM}")
    parser.add_argument("--index", type=Path, help="Defaults to index_DATASET_RawVectors.pkl")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test_file", type=Path, help="Score a saved response CSV; no API key or model call")
    args = parser.parse_args()
    if args.test_file:
        average, patients = binary_metrics(read_csv(args.test_file))
        print(json.dumps({"dataset": args.dataset, "patients": len(patients), "metrics_percent": average}, indent=2))
        return
    if args.data_dir is None or args.out is None:
        parser.error("--data_dir and --out are required for querying")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        parser.error("Set GEMINI_API_KEY in your environment; it is never written to outputs")
    from google import genai
    from google.genai import types
    index_path = args.index or Path("index_" + args.dataset + "_RawVectors.pkl")
    index = load_index(index_path)
    settings = {"dataset": args.dataset, "model": args.model, "temperature": 0, "max_output_tokens": args.max_output_tokens, "M": 2, "prompt_sha256": fingerprint(args.prompt), "index_sha256": fingerprint(index_path), "evaluation_rule": "every seizure + every tenth non-seizure" if args.dataset == "CHSZ" else "every tenth trial of both classes", "limit": args.limit}
    settings_path = args.out.with_suffix(".settings.json")
    if args.out.exists():
        if not args.resume:
            parser.error("Output exists; use --resume or select a new output file")
        if not settings_path.exists() or json.loads(settings_path.read_text()) != settings:
            parser.error("Resume settings differ or are missing; do not combine different experiments")
        done = read_csv(args.out)
        if any(r.get("error") not in ("", "parse-fail") for r in done):
            parser.error("Existing output contains failed requests; archive and resolve them before resuming")
    else:
        done = []
        args.out.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    root = args.data_dir / args.dataset / "time-chan/VLM"
    queries = sorted(root.glob("sub_*/class_*/trial_*_VLM.png"), key=lambda p: key_of(p))
    queries = [p for p in queries if evaluated(key_of(p)[1], key_of(p)[2], args.dataset)]
    if not queries:
        parser.error("No evaluation images found")
    if args.limit:
        queries = queries[:args.limit]
    query_keys = {key_of(p) for p in queries}
    completed = {key_of(row["file_path"]) for row in done}
    if not completed.issubset(query_keys):
        parser.error("Saved output contains trials outside this evaluation list")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=180000))
    try:
        with args.out.open("a" if done else "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            if not done:
                writer.writeheader()
            for query in queries:
                subject, label, _ = key_of(query)
                if key_of(query) in completed:
                    continue
                anchors, seizures, fallback = select_examples(index, subject, str(query), count=2)
                references = [image_path(r["path"]) for r in anchors + seizures]
                text, images = prompt_from_paths(str(query), references, args.prompt)
                parts = []
                import re
                for piece in re.split(r"(\[IMG: [^\]]+\])", text):
                    if piece.startswith("[IMG: "):
                        path = Path(piece[6:-1])
                        if not path.is_file():
                            raise FileNotFoundError("Missing image: " + str(path))
                        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png"))
                    elif piece:
                        parts.append(types.Part.from_text(text=piece))
                response = client.models.generate_content(model=args.model, contents=parts, config=types.GenerateContentConfig(temperature=0, max_output_tokens=args.max_output_tokens))
                raw = response.text
                if not raw:
                    raise RuntimeError("Gemini returned no response text; failed requests are not scored")
                prediction = parse_decision(raw)
                row = {"subject_id": subject, "file_path": str(query), "true_label": label, "pred_label": prediction, "raw_response": raw, "prompt_text": text, "few_shot_paths": repr(references), "few_shot_distances": " | ".join("%s:%.6f" % ("RepNS" if r["label"] == 0 else "NnMedoidSZ", r["dist"]) for r in anchors + seizures), "error": "" if prediction >= 0 else "parse-fail"}
                writer.writerow(row)
                file.flush()
                done.append(row)
                print("%d/%d %s trial %d%s" % (len(done), len(queries), subject, key_of(query)[2], " (later-anchor pool used)" if fallback else ""), flush=True)
    finally:
        client.close()
    average, patients = binary_metrics(done)
    print(json.dumps({"patients": len(patients), "metrics_percent": average}, indent=2))


if __name__ == "__main__":
    main()
