"""
metrics.py — GR-Ben evaluation metrics (no model dependencies).

Functions
---------
predict_sample(scores, threshold) -> int
    Predict the first erroneous step index given per-step scores and a threshold.

calculate_metrics(results, threshold) -> dict
    Compute Acc_Corr, Acc_Err, and F1 over a list of result dicts.

find_optimal_threshold(results) -> (float, float)
    Grid-search the threshold that maximises F1.
"""

from typing import Dict, List, Tuple, Any

import numpy as np


def predict_sample(scores: List[float], threshold: float) -> int:
    """
    Return the 1-indexed position of the first step whose score falls below
    `threshold`, or -1 if all steps pass (i.e., the reasoning is predicted
    correct). Matches the dataset label convention (label=1 means step 1).
    """
    for i, s in enumerate(scores, 1):
        if s < threshold:
            return i
    return -1


def calculate_metrics(results: List[Dict[str, Any]], threshold: float) -> Dict[str, float]:
    """
    Compute step-level evaluation metrics.

    Parameters
    ----------
    results : list of dicts with keys:
        - 'gt_label'  (int): ground-truth first-error step index (-1 = all correct)
        - 'probs'     (list[float]): per-step scores from the PRM
    threshold : float
        Decision threshold; step is predicted erroneous if score < threshold.

    Returns
    -------
    dict with keys 'Acc_Corr', 'Acc_Err', 'F1'
    """
    n_corr = n_err = correct_corr = correct_err = 0

    for res in results:
        gt = res["gt_label"]
        pred = predict_sample(res["probs"], threshold)
        if gt == -1:
            n_corr += 1
            if pred == -1:
                correct_corr += 1
        else:
            n_err += 1
            if pred == gt:
                correct_err += 1

    acc_corr = correct_corr / n_corr if n_corr > 0 else 0.0
    acc_err = correct_err / n_err if n_err > 0 else 0.0
    f1 = 2 * acc_corr * acc_err / (acc_corr + acc_err) if (acc_corr + acc_err) > 0 else 0.0

    return {"Acc_Corr": acc_corr, "Acc_Err": acc_err, "F1": f1}


def find_optimal_threshold(results: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Grid-search the threshold in [min_score, max_score] that maximises F1.

    Parameters
    ----------
    results : list of dicts (same format as `calculate_metrics`)

    Returns
    -------
    (best_threshold, best_f1)
    """
    valid = [r for r in results if r.get("probs")]
    if not valid:
        return 0.5, 0.0

    all_scores = [p for r in valid for p in r["probs"]]
    candidates = np.linspace(
        min(all_scores + [0.0]), max(all_scores + [1.0]), 100
    ).tolist()

    best_f1, best_thresh = -1.0, 0.5
    for thresh in candidates:
        m = calculate_metrics(valid, thresh)
        if m["F1"] > best_f1:
            best_f1, best_thresh = m["F1"], thresh

    return float(best_thresh), best_f1
