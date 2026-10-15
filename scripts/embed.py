"""Extract normalized CLIP ViT-B/32 features and build a patient-indexed lookup.

Reads paired waveform images produced by raicl-render. The benchmark embeds
Vision images without axes or labels; the VLM reads the corresponding VLM images.
Vectors retain float16 storage and their original trial identities.
"""
import argparse
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import re
from pathlib import Path

import numpy as np


# ==========================================
# 1. CUSTOM DATASET FOR BATCH PROCESSING
# ==========================================
class EEGImageDataset:
    def __init__(self, file_paths, preprocess):
        self.file_paths = file_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        from PIL import Image
        path = self.file_paths[idx]
        with Image.open(path) as source:
            image = self.preprocess(source.convert("RGB"))
        return image, str(path)


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
    import clip
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
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
        raise ValueError("No images to embed; do not create an empty retrieval index")

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
            images = images.to(device)
            feats = model.encode_image(images)
            norms = feats.norm(dim=1, keepdim=True)
            if not torch.isfinite(feats).all() or not torch.all(norms > 0):
                raise ValueError("CLIP produced nonfinite or zero embeddings")
            feats = feats / norms

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
    from tqdm import tqdm
    if not Path(embed_dir).exists():
        raise RuntimeError("Embedding directory not found: {}".format(embed_dir))

    embed_dir = Path(embed_dir)

    print("Scanning vectors in: {}".format(embed_dir))
    files = list(embed_dir.rglob("*.npy"))
    print("Found {} embedding files.".format(len(files)))

    if not files:
        raise ValueError("No embeddings found")
    registry = {}
    for f in tqdm(files, desc="Indexing"):
        vec = np.load(f).reshape(-1)
        if vec.shape != (512,) or not np.isfinite(vec).all() or not np.any(vec):
            raise ValueError("Malformed CLIP embedding: " + str(f))
        subjects = [part for part in f.parts if re.fullmatch(r"sub_\d+", part)]
        classes = [part for part in f.parts if part in ("class_0", "class_1")]
        match = re.search(r"trial_(\d+)", f.name)
        img_path = find_existing_image_from_embedding(f)
        if len(subjects) != 1 or len(classes) != 1 or match is None or img_path is None:
            raise ValueError("Embedding has no matching patient, class, trial, or image: " + str(f))
        subject = subjects[0]
        registry.setdefault(subject, []).append({"vec": vec, "label": int(classes[0][-1]), "trial": int(match.group(1)), "path": str(img_path), "embed_path": str(f)})

    with open(output_pkl, "wb") as out:
        pickle.dump(registry, out)

    total = 0
    for k in registry:
        total += len(registry[k])

    print("Saved index to: {}".format(output_pkl))
    print("Subjects: {} | Total items: {}".format(len(registry), total))
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
