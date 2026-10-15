"""Final-decision parsing and patient-macro binary metrics."""
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


DECISION = re.compile(r"FINAL\s+DECISION\s*\**\s*:\s*\**\s*(NON-SEIZURE|SEIZURE)\b", re.I)


def parse_decision(text):
    matches = DECISION.findall((text or "").replace("*", ""))
    return -1 if not matches else int(matches[-1].upper() == "SEIZURE")


def binary_metrics(rows):
    patients = defaultdict(list)
    for row in rows:
        if row.get("error") and row["error"] != "parse-fail":
            raise ValueError("A failed request is not a scored model response")
        true, prediction = int(row["true_label"]), int(row["pred_label"])
        if true not in (0, 1) or prediction not in (-1, 0, 1):
            raise ValueError("Invalid binary label")
        patients[row["subject_id"]].append((true, max(prediction, 0)))
    if not patients:
        raise ValueError("No model responses to score")
    scores = []
    for subject, pairs in sorted(patients.items()):
        true, prediction = zip(*pairs)
        tn, fp, fn, tp = confusion_matrix(true, prediction, labels=[0, 1]).ravel()
        if tp+fn == 0 or tn+fp == 0:
            raise ValueError("Binary sensitivity/specificity undefined for patient " + subject)
        sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)
        scores.append({"patient": subject, "trials": len(pairs), "BCA": 50*(sensitivity+specificity), "Sensitivity": 100*sensitivity, "Specificity": 100*specificity, "Macro F1": 100*f1_score(true, prediction, labels=[0, 1], average="macro", zero_division=0)})
    average = {key: float(np.mean([score[key] for score in scores])) for key in ["BCA", "Sensitivity", "Specificity", "Macro F1"]}
    return average, scores
