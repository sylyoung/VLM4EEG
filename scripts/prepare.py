#!/usr/bin/env python3
"""Prepare CHSZ RAICL inference lists and exact auxiliary-patient fine-tuning queries."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

from raicl import evaluated, image_path, key_of, load_index, prompt_from_paths, select_examples
from raicl.paths import DEFAULT_PROMPT, TRAINING_MANIFEST


def references(index, query, excluded=()):
    anchors, seizures, fallback = select_examples(index, query["key"][0], query["path"], excluded=excluded)
    paths = [image_path(item["path"]) for item in anchors + seizures]
    return paths, " | ".join("%s:%.6f" % ("RepNS" if item["label"] == 0 else "NnMedoidSZ", item["dist"]) for item in anchors+seizures), fallback


def prepare(args):
    index = load_index(args.index)
    by_key = {item["key"]: item for entries in index.values() for item in entries}
    if args.subject not in index:
        raise ValueError("Held-out patient is absent from the index")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    test_file = args.out_dir / (args.subject + "_raicl.csv")
    if test_file.exists():
        raise FileExistsError("Do not overwrite existing trial/reference lists")
    rows = []
    for query in sorted(index[args.subject], key=lambda item: item["trial"]):
        if not evaluated(query["label"], query["trial"], "CHSZ"):
            continue
        paths, distances, fallback = references(index, query)
        rows.append({"subject_id": args.subject, "file_path": image_path(query["path"]), "true_label": query["label"], "few_shot_paths": repr(paths), "few_shot_distances": distances, "later_anchor_fallback": int(fallback)})
    if not rows:
        raise ValueError("Held-out patient has no evaluation trials")
    if args.training:
        manifest = json.loads(args.training_manifest.read_text())
        ordered = manifest[args.subject]
        training, targets = [], []
        for name in ordered:
            query_key = key_of(name + "_VLM.png")
            if query_key not in by_key or query_key[0] == args.subject:
                raise ValueError("Training query is missing or belongs to the held-out patient")
            query = by_key[query_key]
            if query_key[1] == 0 and evaluated(0, query_key[2], "CHSZ"):
                raise ValueError("Training non-seizure query appears on the evaluation list")
            paths, _, _ = references(index, query, excluded=(args.subject,))
            text, images = prompt_from_paths(image_path(query["path"]), paths, args.prompt)
            if any(key_of(path)[0] == args.subject for path in images):
                raise ValueError("Held-out patient image leaked into fine-tuning")
            if any(not Path(path).is_file() for path in images):
                raise FileNotFoundError("A training query or reference image is absent")
            target = "FINAL DECISION: " + ("SEIZURE" if query_key[1] else "NON-SEIZURE")
            training.append({"messages": [{"role": "user", "content": re.sub(r"\[IMG: [^\]]+\]", "<image>", text)}, {"role": "assistant", "content": target}], "images": images})
            targets.append(query_key[1])
        if targets.count(0) != targets.count(1):
            raise ValueError("Fine-tuning queries are not class-balanced")
        data_name = "chsz_raicl_loso_" + args.subject
        training_file = args.out_dir / (data_name + ".jsonl")
        info_file = args.out_dir / "dataset_info.json"
        config_file = args.out_dir / (data_name + ".yaml")
        if any(path.exists() for path in (training_file, info_file, config_file)):
            raise FileExistsError("Use an empty preparation directory for each fold")
        training_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in training))
        info = {data_name: {"file_name": training_file.name, "formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"}, "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}}
        info_file.write_text(json.dumps(info, indent=2) + "\n")
        steps = math.ceil(len(training)/16)
        config = {"model_name_or_path": args.model, "template": "qwen3_5", "stage": "sft", "do_train": True, "do_eval": False, "finetuning_type": "lora", "quantization_bit": 4, "double_quantization": True, "quantization_method": "bnb", "dataset": data_name, "dataset_dir": str(args.out_dir.resolve()), "cutoff_len": 4096, "overwrite_cache": True, "lora_rank": 32, "lora_alpha": 16, "lora_dropout": 0.05, "freeze_vision_tower": False, "freeze_multi_modal_projector": False, "freeze_language_model": False, "lora_target": ["qkv", "proj", "linear_fc1", "linear_fc2", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "learning_rate": 1.0e-4, "num_train_epochs": 2.0, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16, "lr_scheduler_type": "cosine", "warmup_ratio": 0.05, "bf16": True, "gradient_checkpointing": True, "dataloader_num_workers": 0, "logging_steps": 5, "save_steps": max(1, steps//2), "save_total_limit": 4, "output_dir": str((args.out_dir / "adapter").resolve()), "seed": 42}
        import yaml
        config_file.write_text(yaml.safe_dump(config, sort_keys=False))
        (args.out_dir / "protocol.json").write_text(json.dumps({"held_out_patient": args.subject, "training_queries": len(training), "queries_per_class": len(training)//2, "M": 2, "epochs": 2, "optimizer_steps": 2*steps, "checkpoint_selection": "final epoch; no held-out-patient selection", "test_queries": len(rows)}, indent=2)+"\n")
        print(config_file)
        print("Final adapter: %s (%d optimizer steps)" % (config["output_dir"], 2*steps))
    with test_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Prepared", len(rows), "test queries for", args.subject)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--training-manifest", type=Path, default=TRAINING_MANIFEST)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
