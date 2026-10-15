#!/usr/bin/env python3
"""Run a held-out patient's Qwen3.5-9B adapter with the saved RAICL references."""
import argparse
import ast
import csv
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from raicl import FIELDS, binary_metrics, key_of, parse_decision, prompt_from_paths, read_csv
from raicl.paths import DEFAULT_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen/Qwen3.5-9B or a local model directory")
    parser.add_argument("--adapter", required=True, help="Final LoRA checkpoint for this held-out patient")
    parser.add_argument("--csv", type=Path, required=True, help="One patient's query/reference CSV from raicl-prepare")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("Batch size and token budget must be positive")
    rows = read_csv(args.csv)
    if not rows or len({r["subject_id"] for r in rows}) != 1:
        parser.error("Use exactly one held-out patient with its own trained adapter")
    settings = {"model": args.model, "adapter": str(Path(args.adapter).resolve()), "input_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(), "prompt_sha256": hashlib.sha256(args.prompt.read_bytes()).hexdigest(), "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "do_sample": False, "quantization_bit": 4, "template": "qwen3_5"}
    settings_path = args.out.with_suffix(".settings.json")
    if args.out.exists():
        if not args.resume or not settings_path.exists() or json.loads(settings_path.read_text()) != settings:
            parser.error("Existing output cannot be overwritten or resumed with different settings")
        done = read_csv(args.out)
        if any(r.get("error") not in ("", "parse-fail") for r in done):
            parser.error("Output contains a generation failure, not a measured response")
    else:
        done = []
        args.out.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    expected = {key_of(r["file_path"]) for r in rows}
    completed = {key_of(r["file_path"]) for r in done}
    if not completed.issubset(expected):
        parser.error("Existing output contains trials outside the selected patient")
    pending = [r for r in rows if key_of(r["file_path"]) not in completed]
    if pending:
        import torch
        from transformers import BitsAndBytesConfig
        original = BitsAndBytesConfig.__init__
        def uint8_storage(self, **kwargs):
            if kwargs.get("load_in_4bit"):
                kwargs["bnb_4bit_quant_storage"] = torch.uint8
            original(self, **kwargs)
        BitsAndBytesConfig.__init__ = uint8_storage
        from llamafactory.chat import ChatModel
        from llamafactory.extras.constants import IMAGE_PLACEHOLDER
        from raicl.models.qwen_batch_inputs import assemble_batch, process_one
        chat = ChatModel({"model_name_or_path": args.model, "adapter_name_or_path": args.adapter, "template": "qwen3_5", "infer_backend": "huggingface", "quantization_bit": 4, "quantization_method": "bnb", "double_quantization": True})
        engine = chat.engine
        with args.out.open("a" if done else "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            if not done:
                writer.writeheader()
            for offset in range(0, len(pending), args.batch_size):
                chunk = pending[offset:offset+args.batch_size]
                inputs, prompts, references = [], [], []
                for row in chunk:
                    refs = ast.literal_eval(row["few_shot_paths"])
                    subject = row["subject_id"]
                    if len(refs) != 4 or any(key_of(p)[0] != subject or key_of(p)[1] != 0 for p in refs[:2]) or any(key_of(p)[0] == subject or key_of(p)[1] != 1 for p in refs[2:]):
                        raise ValueError("References violate the patient-specific RAICL pool rules")
                    if any(key_of(p) == key_of(row["file_path"]) for p in refs):
                        raise ValueError("A query cannot be its own reference")
                    prompt, images = prompt_from_paths(row["file_path"], refs, args.prompt)
                    if any(not Path(path).is_file() for path in images):
                        raise FileNotFoundError("A query or reference image is missing")
                    content = re.sub(r"\[IMG: [^\]]+\]", IMAGE_PLACEHOLDER, prompt)
                    generated, _ = process_one(engine, [{"role": "user", "content": content}], images, {"max_new_tokens": args.max_new_tokens, "do_sample": False, "temperature": 0.0})
                    inputs.append(generated)
                    prompts.append(prompt)
                    references.append(refs)
                batch = assemble_batch(inputs, engine.tokenizer)
                prompt_length = batch["input_ids"].shape[1]
                with torch.inference_mode():
                    output = engine.model.generate(**batch)
                # Each generated sequence includes the full padded input, not its unpadded length.
                responses = engine.tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                if len(responses) != len(chunk):
                    raise ValueError("Generation returned the wrong number of sequences")
                for row, prompt, refs, raw in zip(chunk, prompts, references, responses):
                    if not raw:
                        raise RuntimeError("No model response; do not record a failed generation as a prediction")
                    prediction = parse_decision(raw)
                    record = {"subject_id": row["subject_id"], "file_path": row["file_path"], "true_label": row["true_label"], "pred_label": prediction, "raw_response": raw, "prompt_text": prompt, "few_shot_paths": repr(refs), "few_shot_distances": row.get("few_shot_distances", ""), "error": "" if prediction >= 0 else "parse-fail"}
                    writer.writerow(record)
                    done.append(record)
                file.flush()
                print("%d/%d trials completed" % (len(done), len(rows)), flush=True)
    average, _ = binary_metrics(done)
    print(json.dumps({"metrics_percent": average}, indent=2))


if __name__ == "__main__":
    main()
