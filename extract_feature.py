"""
Filename: extract_feature.py
Author: Siyang Li
Date: 2026-01-23
Description: Use CLIP's visual encoder for feature extraction of EEG stacked waveform images.
             Requires running plot_timechan.py first
Pipeline:
1) Extract CLIP (ViT-B/32) normalized embeddings from images under:
   {data_dir}/{dataset}/time-chan/{image_type}/
2) Save embeddings to:
   {data_dir}/{dataset}/time-chan/{image_type}_Embeddings/
3) Build a pickle index from:
   {data_dir}/{dataset}/time-chan/{index_from}_Embeddings/
The .pkl file saved to ./index_{dataset}_RawVectors.pkl will be used in query API for RAICL selection algorithm.
Note that we used features from "Vision"-type of plot, without the axis/ticks/channels-names of "VLM"-type of plot in the paper.
It is possible that using "VLM" type can be better for larger VLMs' visual encoder instead of CLIP's visual encoder.
"""
import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ==========================================
# 1. CUSTOM DATASET FOR BATCH PROCESSING
# ==========================================
class EEGImageDataset(Dataset):
    def __init__(self, file_paths, preprocess):
        self.file_paths = file_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            image = self.preprocess(image)
            return image, str(path)
        except Exception as e:
            print("[WARN] Error loading {}: {}".format(path, e))
            # dummy tensor, will be filtered out
            return torch.zeros(3, 224, 224), "ERROR"


# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
def extract_clip_features(
    data_dir,
    dataset,
    image_type,
    batch_size,
    num_workers,
    model_name="ViT-B/32",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))

    print("Loading CLIP model: {} ...".format(model_name))
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    input_base = Path(data_dir) / dataset / "time-chan" / image_type
    output_base = Path(data_dir) / dataset / "time-chan" / (image_type + "_Embeddings")

    print("Input Directory:  {}".format(input_base))
    print("Output Directory: {}".format(output_base))

    if not input_base.exists():
        raise RuntimeError("Input directory does not exist: {}".format(input_base))

    print("Scanning images...")
    all_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        all_files.extend(list(input_base.rglob(ext)))

    print("Found {} images.".format(len(all_files)))
    if len(all_files) == 0:
        return input_base, output_base, 0

    dataset_obj = EEGImageDataset(all_files, preprocess)
    dataloader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    print("Starting CLIP feature extraction...")
    saved = 0
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting Features"):
            # filter out errors
            valid_mask = [p != "ERROR" for p in paths]
            if not any(valid_mask):
                continue

            images = images[valid_mask].to(device)
            paths = [p for i, p in enumerate(paths) if valid_mask[i]]

            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=1, keepdim=True)

            feats_np = feats.detach().cpu().numpy().astype(np.float16)

            for i, original_path in enumerate(paths):
                orig_path_obj = Path(original_path)
                rel = orig_path_obj.relative_to(input_base)
                save_path = output_base / rel.with_suffix(".npy")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(save_path, feats_np[i])
                saved += 1

    print("Extraction complete. Saved {} vectors.".format(saved))
    return input_base, output_base, saved


# ==========================================
# 3. INDEX BUILDING
# ==========================================
def find_existing_image_from_embedding(embedding_path):
    """
    Map:
      .../{type}_Embeddings/.../trial_1.npy
    to:
      .../{type}/.../trial_1.{png|jpg|jpeg} (whichever exists)
    """
    # Remove "_Embeddings" from the path string and strip .npy
    img_stem = Path(str(embedding_path).replace("_Embeddings", "")).with_suffix("")
    for suf in (".png", ".jpg", ".jpeg"):
        candidate = img_stem.with_suffix(suf)
        if candidate.exists():
            return candidate
    return None


def build_index_from_embeddings(embed_dir, output_pkl, subject_prefixes=("sub", "chb", "pat"), class1_token="class_1"):
    if not Path(embed_dir).exists():
        raise RuntimeError("Embedding directory not found: {}".format(embed_dir))

    embed_dir = Path(embed_dir)

    print("Scanning vectors in: {}".format(embed_dir))
    files = list(embed_dir.rglob("*.npy"))
    print("Found {} embedding files.".format(len(files)))

    registry = {}
    skipped = 0

    for f in tqdm(files, desc="Indexing"):
        try:
            vec = np.load(f).reshape(-1)

            parts = f.parts
            sub = "unknown"
            for p in parts:
                for pref in subject_prefixes:
                    if p.startswith(pref):
                        sub = p
                        break
                if sub != "unknown":
                    break

            lbl = 1 if class1_token in parts else 0

            t_match = re.search(r"trial_(\d+)", f.name)
            trial = int(t_match.group(1)) if t_match else -1

            img_path = find_existing_image_from_embedding(f)
            img_path_str = str(img_path) if img_path is not None else None

            if sub not in registry:
                registry[sub] = []

            registry[sub].append(
                {
                    "vec": vec,
                    "label": lbl,
                    "trial": trial,
                    "path": img_path_str,
                    "embed_path": str(f),
                }
            )
        except Exception as e:
            skipped += 1
            print("[WARN] Skipping {}: {}".format(f, e))

    with open(output_pkl, "wb") as out:
        pickle.dump(registry, out)

    total = 0
    for k in registry:
        total += len(registry[k])

    print("Saved index to: {}".format(output_pkl))
    print("Subjects: {} | Total items: {} | Skipped: {}".format(len(registry), total, skipped))
    return registry


# ==========================================
# 4. MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features of EEG waveform images and build an index.")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/",
        help="Root directory containing dataset folders",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., CHSZ or NICU)")
    parser.add_argument("--image_type", type=str, default="Vision", choices=["Vision", "VLM"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--index_from",
        type=str,
        default="Vision",
        choices=["Vision", "VLM"],
        help="Which embedding directory to index from",
    )
    parser.add_argument(
        "--index_out",
        type=str,
        default=None,
        help="Output pickle path. Default: ./index_{dataset}_RawVectors.pkl",
    )

    args = parser.parse_args()

    if args.image_type == "VLM":
        print("[NOTE] You selected 'VLM' (images with text/legends). CLIP may encode legend/text content.")

    # 1) Extract features from args.image_type
    extract_clip_features(
        data_dir=args.data_dir,
        dataset=args.dataset,
        image_type=args.image_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 2) Build index from args.index_from (default Vision_Embeddings)
    embed_dir = Path(args.data_dir) / args.dataset / "time-chan" / (args.index_from + "_Embeddings")
    output_pkl = args.index_out if args.index_out else "./index_{}_RawVectors.pkl".format(args.dataset)

    build_index_from_embeddings(embed_dir=embed_dir, output_pkl=output_pkl)


if __name__ == "__main__":
    main()
