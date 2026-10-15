"""Try the RAICL retrieval rule on synthetic features, without EEG, a GPU or an API."""
import argparse

import numpy as np

from raicl import prompt_from_paths, select_examples


def entry(subject, label, trial, values):
    vector = np.zeros(512, dtype=np.float16)
    vector[:3] = values
    return {
        "path": f"data/CHSZ/time-chan/Vision/{subject}/class_{label}/trial_{trial}_Vision.png",
        "label": label,
        "trial": trial,
        "key": (subject, label, trial),
        "vec": vector,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    index = {
        "sub_00": [entry("sub_00", 0, 10, [1, 1, 0]),
                   entry("sub_00", 0, 20, [1, 0.9, 0]),
                   entry("sub_00", 1, 30, [0, 1, 1])],
        "sub_01": [entry("sub_01", 1, 11, [0, 1, 0.9])],
        "sub_02": [entry("sub_02", 1, 40, [0, 1, 0.8])],
        "sub_03": [entry("sub_03", 1, 50, [0, 0.8, 1])],
    }
    query = index["sub_00"][-1]["path"]
    anchors, seizures, fallback = select_examples(index, "sub_00", query)
    references = [item["path"] for item in anchors + seizures]
    prompt, images = prompt_from_paths(query, references)
    print("Synthetic features only; no EEG images are loaded and no model is queried.\n")
    print("Same-patient non-seizure anchors:")
    for item in anchors:
        print(" ", item["key"])
    print("Similar seizure medoids from other patients:")
    for item in seizures:
        print(" ", item["key"])
    print("Later-anchor pool used:", fallback)
    print("Prompt image order:")
    for position, path in enumerate(images, 1):
        print(f"  {position}. {path}")
    print(f"Assembled {len(images)} image slots and {len(prompt)} text characters.")


if __name__ == "__main__":
    main()
