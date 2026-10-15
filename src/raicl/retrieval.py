"""Patient-aware representative and similarity-based retrieval."""
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances


KEY = re.compile(r"(sub_\d+)/class_([01])/trial_(\d+)_(?:VLM|Vision)\.png$")


def key_of(path):
    match = KEY.search(str(path).replace("\\", "/"))
    if not match:
        raise ValueError("Expected sub_XX/class_Y/trial_N_VLM.png: " + str(path))
    return match.group(1), int(match.group(2)), int(match.group(3))


def image_path(path, image_type="VLM"):
    return re.sub(r"/(?:Vision|VLM)/", "/" + image_type + "/", str(path).replace("\\", "/")).replace("_Vision.png", "_" + image_type + ".png").replace("_VLM.png", "_" + image_type + ".png")


def evaluated(label, trial, dataset):
    if dataset not in ("CHSZ", "NICU"):
        raise ValueError("Unknown dataset: " + dataset)
    return trial % 10 == 0 or (dataset == "CHSZ" and label == 1)


def load_index(path):
    # Pickle must come from your own feature extractor, never an untrusted download.
    with Path(path).open("rb") as file:
        index = pickle.load(file)
    if not isinstance(index, dict) or not index:
        raise ValueError("The retrieval index is empty or malformed")
    seen = set()
    for subject, entries in index.items():
        for item in entries:
            key = key_of(item["path"])
            if key in seen or key[0] != subject or key[1] != item["label"] or key[2] != item["trial"]:
                raise ValueError("Inconsistent/duplicate index entry: " + str(key))
            vector = np.asarray(item["vec"])
            if vector.shape != (512,) or not np.isfinite(vector).all() or not np.any(vector):
                raise ValueError("CLIP ViT-B/32 requires a finite nonzero 512-dimensional vector")
            item["key"] = key
            seen.add(key)
    return index


def nearest(pool, center, count):
    if not pool or count <= 0:
        raise ValueError("Reference pool is empty or reference count is not positive")
    vectors = np.stack([item["vec"] for item in pool])
    distances = cosine_distances(vectors, np.asarray(center).reshape(1, -1)).ravel()
    result = [dict(pool[i], dist=float(distances[i])) for i in np.argsort(distances)[:count]]
    # This is the original rule for a short non-seizure pool; it never uses the query.
    while len(result) < count:
        result.append(result[0].copy())
    return result


def select_examples(index, subject, test_path, count=2, excluded=()):
    query_key = key_of(test_path)
    if subject != query_key[0] or subject not in index:
        raise ValueError("Query patient does not match its retrieval index")
    if subject in excluded:
        raise ValueError("An excluded patient cannot supply a training query")
    query = next((item for item in index[subject] if item["key"] == query_key), None)
    if query is None:
        raise ValueError("Query embedding is absent from the retrieval index")
    pool = [item for item in index[subject] if item["label"] == 0 and item["trial"] < query_key[2]]
    later_fallback = len(pool) < count
    if later_fallback:
        pool.extend(item for item in index[subject] if item["label"] == 0 and item["trial"] > query_key[2])
    if not pool:
        raise ValueError("Patient has no non-seizure anchor other than the query")
    anchors = nearest(pool, np.mean(np.stack([item["vec"] for item in pool]), axis=0), count)
    medoids = []
    for other, entries in index.items():
        if other == subject or other in excluded:
            continue
        seizures = [item for item in entries if item["label"] == 1]
        if seizures:
            center = np.mean(np.stack([item["vec"] for item in seizures]), axis=0)
            medoids.extend(nearest(seizures, center, 1))
    if len(medoids) < count:
        raise ValueError("Too few auxiliary patients with seizure examples")
    seizures = nearest(medoids, query["vec"], count)
    if any(item["key"] == query_key for item in anchors + seizures):
        raise ValueError("Query was selected as its own reference")
    return anchors, seizures, later_fallback
