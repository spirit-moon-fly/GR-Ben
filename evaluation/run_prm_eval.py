"""
run_prm_eval.py — Evaluate a PRM on GR-Ben.

Two modes:

1. Inference + metrics (Qwen models built-in):

    python evaluation/run_prm_eval.py infer \
        --data data/GR-BEN.jsonl \
        --model_type qwen \
        --model_path /path/to/Qwen2.5-Math-PRM-7B \
        --output_dir results/qwen_run

2. Metrics only (from pre-computed results):

    python evaluation/run_prm_eval.py eval \
        --input_file results/my_model_results.jsonl \
        --threshold 0.5

    Input format (one JSON object per line):
        {"id": "physics-42", "domain": "physics",
         "gt_label": 3, "probs": [0.91, 0.87, 0.82, 0.31]}
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from metrics import calculate_metrics, find_optimal_threshold, predict_sample


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_cached(path: str):
    if not os.path.exists(path):
        return [], set()
    records = load_jsonl(path)
    return records, {r.get("id", "") for r in records}


def macro_average(rows: List[Dict]) -> Dict:
    """Macro-average of per-domain Acc_Corr, Acc_Err, and F1."""
    if not rows:
        return {"Acc_Corr": 0.0, "Acc_Err": 0.0, "F1": 0.0}
    n = len(rows)
    return {
        "Acc_Corr": sum(r["Acc_Corr"] for r in rows) / n,
        "Acc_Err":  sum(r["Acc_Err"]  for r in rows) / n,
        "F1":       sum(r["F1"]       for r in rows) / n,
    }


def print_domain_table(rows: List[Dict], overall: Dict, n_total: int):
    print(f"\n{'Domain':<25} {'F1':>8} {'Acc_Corr':>10} {'Acc_Err':>10} {'N':>6}")
    print("-" * 65)
    for row in rows:
        m = row
        print(
            f"{m['Domain']:<25} {m['F1']*100:>7.2f}%"
            f" {m['Acc_Corr']*100:>9.2f}%"
            f" {m['Acc_Err']*100:>9.2f}%"
            f" {m['N']:>6}"
        )
    print("-" * 65)
    print(
        f"{'Average':<25} {overall['F1']*100:>7.2f}%"
        f" {overall['Acc_Corr']*100:>9.2f}%"
        f" {overall['Acc_Err']*100:>9.2f}%"
        f" {n_total:>6}"
    )


# ---------------------------------------------------------------------------
# Mode 1: Inference with built-in Qwen adapter
# ---------------------------------------------------------------------------

def run_inference(adapter, items, output_file, threshold, desc, processed_ids):
    to_run = [it for it in items if it.get("id") not in processed_ids]
    if not to_run:
        return []

    if adapter.model is None:
        adapter.load()

    results = []
    with open(output_file, "a", encoding="utf-8") as fout:
        for item in tqdm(to_run, desc=desc, leave=False):
            query = item.get("problem", item.get("question", ""))
            steps = item["steps"]
            gt_label = int(item["label"])
            if gt_label != -1 and gt_label > len(steps):
                gt_label = len(steps)

            probs = adapter.score_steps(query, steps)
            pred_label = predict_sample(probs, threshold)
            match = gt_label == pred_label

            rec = {
                "id": item.get("id", ""),
                "domain": item.get("domain", ""),
                "generator": item.get("generator", ""),
                "gt_label": gt_label,
                "pred_label": pred_label,
                "threshold": threshold,
                "probs": probs,
                "is_correct_prediction": match,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            results.append(rec)
    return results


def cmd_infer(args):
    from models import build_adapter

    import torch
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model_key = args.model_key or os.path.basename(args.model_path.rstrip("/\\"))
    threshold = args.threshold if args.threshold is not None else 0.5

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading data from {args.data} ...")
    all_items = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    domain_items: Dict[str, List] = defaultdict(list)
    for item in all_items:
        domain_items[item["domain"]].append(item)
    domains = sorted(domain_items.keys())
    print(f"Loaded {len(all_items)} items across {len(domains)} domains.")

    adapter = build_adapter(args.model_type, model_key, args.model_path, device)

    print(f"\nModel  : {model_key}  |  Threshold: {threshold}")
    print("=" * 70)

    summary_rows = []
    for domain in domains:
        items_full = domain_items[domain]
        domain_dir = os.path.join(args.output_dir, domain, "prm_result")
        os.makedirs(domain_dir, exist_ok=True)
        detail_file = os.path.join(domain_dir, f"{model_key}_details.jsonl")

        cached, processed_ids = load_cached(detail_file)
        new_results = run_inference(
            adapter, items_full, detail_file, threshold,
            desc=f"Eval:{domain}", processed_ids=processed_ids,
        )
        all_results = cached + new_results
        m = calculate_metrics(all_results, threshold)
        summary_rows.append({"Domain": domain, "N": len(all_results), **m})

    if adapter.model is not None:
        adapter.unload()

    all_records = [r for rows in [load_jsonl(
        os.path.join(args.output_dir, d, "prm_result", f"{model_key}_details.jsonl")
    ) for d in domains] for r in rows]
    overall = macro_average(summary_rows)

    print_domain_table(summary_rows, overall, len(all_records))

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.output_dir, f"{model_key}_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Mode 2: Metrics from pre-computed results
# ---------------------------------------------------------------------------

def cmd_eval(args):
    if args.input_file:
        all_records = load_jsonl(args.input_file)
    else:
        all_records = []
        for path in glob.glob(os.path.join(args.input_dir, "**", "*.jsonl"), recursive=True):
            all_records.extend(load_jsonl(path))

    if not all_records:
        print("No records found.")
        return

    print(f"Loaded {len(all_records)} records.")

    if args.tune_threshold:
        threshold, best_f1 = find_optimal_threshold(all_records)
        print(f"Tuned threshold: {threshold:.4f}  (F1={best_f1*100:.2f}%)")
    else:
        threshold = args.threshold

    by_domain: Dict[str, List] = defaultdict(list)
    for r in all_records:
        by_domain[r.get("domain", "unknown")].append(r)

    rows = []
    for domain in sorted(by_domain.keys()):
        recs = by_domain[domain]
        m = calculate_metrics(recs, threshold)
        rows.append({"Domain": domain, "N": len(recs), **m})

    overall = macro_average(rows)
    print_domain_table(rows, overall, len(all_records))

    if args.output:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\nSummary saved to: {args.output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GR-Ben PRM evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- infer ---
    p_infer = sub.add_parser("infer", help="Run Qwen PRM inference + compute metrics.")
    p_infer.add_argument("--data", required=True, help="Path to GR-BEN.jsonl")
    p_infer.add_argument("--model_type", required=True, choices=["qwen"],
                         help="Adapter type")
    p_infer.add_argument("--model_path", required=True, help="Path to model weights")
    p_infer.add_argument("--output_dir", required=True, help="Output directory")
    p_infer.add_argument("--model_key", default=None, help="Display name in output files")
    p_infer.add_argument("--device", default=None, help="Torch device (e.g. cuda:0)")
    p_infer.add_argument("--threshold", type=float, default=None,
                         help="Decision threshold (default: 0.5)")

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Compute metrics from pre-computed results.")
    grp = p_eval.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input_file", help="Single merged JSONL result file")
    grp.add_argument("--input_dir", help="Directory containing result JSONL files")
    p_eval.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (default: 0.5)")
    p_eval.add_argument("--tune_threshold", action="store_true",
                        help="Grid-search best threshold instead of using --threshold")
    p_eval.add_argument("--output", default=None, help="Save summary CSV to this path")

    args = parser.parse_args()
    if args.mode == "infer":
        cmd_infer(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
