"""
run_llm_eval.py — Evaluate an LLM on GR-Ben via API.

Three subcommands:

1. infer  — Call a single model API for every item in GR-BEN.jsonl:

    python evaluation/run_llm_eval.py infer \\
        --data data/GR-BEN.jsonl \\
        --model gpt-4o \\
        --api_key $OPENAI_KEY \\
        --base_url https://api.openai.com/v1/chat/completions \\
        --output_dir results/gpt4o \\
        --mode Fast

   Thinking-mode variants (--mode Slow):
     --thinking_type reasoning_effort   OpenAI o1/o3/gpt-5 style
     --thinking_type enable_thinking    DeepSeek native style
     --thinking_type budget_tokens      Claude style (--budget_tokens 8192)

2. batch  — Run multiple model configs from a JSON file or a builtin preset:

    python evaluation/run_llm_eval.py batch \\
        --data data/GR-BEN.jsonl \\
        --preset gemini \\
        --api_key $API_KEY \\
        --output_dir results/

    python evaluation/run_llm_eval.py batch \\
        --data data/GR-BEN.jsonl \\
        --config my_models.json \\
        --output_dir results/

   Builtin presets: gemini

   Config file format (JSON array):
    [
      {
        "model":        "gemini-3-flash-preview-nothinking",
        "model_key":    "Gemini-3-flash",
        "api_key":      "YOUR_KEY",
        "base_url":     "https://api.example.com/v1/chat/completions",
        "mode":         "Fast",
        "thinking_type": null
      },
      ...
    ]

3. eval   — Compute metrics from saved prediction files:

    python evaluation/run_llm_eval.py eval \\
        --input_file results/gpt4o/biology/llm_result/gpt-4o_Fast.jsonl

    python evaluation/run_llm_eval.py eval \\
        --input_dir results/gpt4o

Output layout:
    output_dir/{domain}/llm_result/{model_key}_{mode}.jsonl   per-domain detail
    output_dir/{model_key}_{mode}_summary.csv                 per-domain metrics
    output_dir/model_comparison_results.jsonl                 aggregated summary
"""

import argparse
import asyncio
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio


# ---------------------------------------------------------------------------
# Builtin batch presets
#
# Gemini uses *separate model names* to switch between fast and slow thinking
# (gemini-3-flash-preview-nothinking vs. gemini-3-flash-preview-thinking),
# so thinking_type is None and the mode is baked into the model name itself.
# Replace "YOUR_API_KEY" with your actual key, or pass --api_key on the CLI.
# ---------------------------------------------------------------------------

BUILTIN_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "gpt5": [
        {
            "model":        "gpt-5.2-2025-12-11",
            "model_key":    "GPT-5.2",
            "api_key":      "YOUR_API_KEY",
            "base_url":     "https://api.openai.com/v1/chat/completions",
            "mode":         "Fast",
            "thinking_type": "reasoning_effort",
            "max_tokens":   32768,
            "concurrency":  10,
        },
        {
            "model":        "gpt-5.2-2025-12-11",
            "model_key":    "GPT-5.2",
            "api_key":      "YOUR_API_KEY",
            "base_url":     "https://api.openai.com/v1/chat/completions",
            "mode":         "Slow",
            "thinking_type": "reasoning_effort",
            "max_tokens":   32768,
            "concurrency":  10,
        },
    ],
    "gemini": [
        {
            "model":        "gemini-3-flash-preview-nothinking",
            "model_key":    "Gemini-3-flash",
            "api_key":      "YOUR_API_KEY",
            "base_url":     "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "mode":         "Fast",
            "thinking_type": None,
            "max_tokens":   32768,
            "concurrency":  30,
        },
        {
            "model":        "gemini-3-flash-preview-thinking",
            "model_key":    "Gemini-3-flash",
            "api_key":      "YOUR_API_KEY",
            "base_url":     "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "mode":         "Slow",
            "thinking_type": None,
            "max_tokens":   32768,
            "concurrency":  30,
        },
    ],
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def construct_prompt(item: Dict[str, Any]) -> str:
    problem = item.get("problem", item.get("question", ""))
    steps = item.get("steps", [])
    tagged = "\n".join(f"<paragraph_{i}> {s} </paragraph_{i}>" for i, s in enumerate(steps, 1))
    return (
        "The following is a problem and a solution "
        "(split into paragraphs, enclosed with tags and indexed from 1):\n\n"
        f"[Problem]\n{problem}\n\n"
        f"[Solution]\n{tagged}\n\n"
        "Your task is to review and critique the solution paragraph by paragraph. "
        "Once you identify an error in a paragraph, return the index of the paragraph "
        "where the earliest error occurs. "
        "Otherwise, return the index of -1 (which typically denotes \"not found\").\n"
        "Please put your final answer (i.e., the index) in \\boxed{}."
    )


def extract_boxed_answer(text: str) -> int:
    """Parse the last \\boxed{N} integer from the model response."""
    if not text:
        return -999
    matches = re.findall(r"\\boxed\s*\{\s*(-?\d+)\s*\}", text)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            pass
    return -999


# ---------------------------------------------------------------------------
# Async API call
# ---------------------------------------------------------------------------

async def call_api(
    session: aiohttp.ClientSession,
    item: Dict[str, Any],
    model: str,
    api_key: str,
    base_url: str,
    mode: str,
    thinking_type: Optional[str],
    budget_tokens: int,
    max_tokens: int,
    max_retries: int,
    backoff_cap: float,
) -> Dict[str, Any]:
    """
    Call the chat completion API for one item and return the annotated result dict.

    On unrecoverable failure, returns the original item with api_failed=True.
    """
    user_content = construct_prompt(item)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict reasoning process auditor."},
            {"role": "user",   "content": user_content},
        ],
    }

    # --- thinking / token configuration ---
    if thinking_type == "reasoning_effort":
        # OpenAI o1 / o3 / gpt-5 style
        payload["max_completion_tokens"] = max_tokens
        payload["reasoning_effort"] = "none" if mode == "Fast" else "high"

    elif thinking_type == "enable_thinking":
        # DeepSeek native thinking style
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0.6
        if mode == "Slow":
            payload["enable_thinking"] = True

    elif thinking_type == "budget_tokens":
        # Claude extended-thinking style
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0.6
        if mode == "Slow":
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        else:
            payload["thinking"] = {"type": "disabled"}

    else:
        # Standard model (no special thinking params)
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0.6

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            async with session.post(
                base_url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = ""
                    if data.get("choices"):
                        content = data["choices"][0]["message"].get("content", "")
                    pred = extract_boxed_answer(content)
                    result = item.copy()
                    result.update({
                        "pred_error_idx": pred,
                        "raw_response":   content,
                        "thinking_mode":  mode,
                        "model_key":      model,
                        "api_failed":     False,
                    })
                    return result

                elif resp.status == 400:
                    # Bad request — retrying won't help
                    print(f"  [400 Bad Request] model={model} id={item.get('id')}: {await resp.text()[:200]}")
                    break

                elif resp.status in (429, 500, 502, 503, 504):
                    pass  # transient; fall through to retry

                else:
                    print(f"  [HTTP {resp.status}] model={model} — giving up on this item.")
                    break

        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass

        # Exponential backoff
        if attempt < max_retries - 1:
            wait = min(backoff_cap, (2 ** attempt) + random.uniform(0, 1))
            if wait > 15:
                print(f"  Backing off {wait:.1f}s (attempt {attempt + 1}/{max_retries}) …")
            await asyncio.sleep(wait)

    # All retries exhausted
    fallback = item.copy()
    fallback.update({"pred_error_idx": -999, "api_failed": True,
                     "thinking_mode": mode, "model_key": model})
    return fallback


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute Acc_Corr, Acc_Err, and F1.

    A prediction is correct when:
      - gt_label == -1  and  pred_error_idx == -1   (correctly identified as clean)
      - gt_label >= 0   and  pred_error_idx == gt_label  (correctly found the error step)

    Samples with api_failed=True or pred_error_idx == -999 are excluded.
    """
    n_corr = n_err = hit_corr = hit_err = 0
    for r in records:
        if r.get("api_failed") or r.get("pred_error_idx") == -999:
            continue
        gt   = r.get("label", r.get("gt_label"))
        pred = r.get("pred_error_idx")
        if gt is None:
            continue
        try:
            gt, pred = int(gt), int(pred)
        except (TypeError, ValueError):
            continue

        if gt == -1:
            n_corr += 1
            if pred == -1:
                hit_corr += 1
        else:
            n_err += 1
            if pred == gt:
                hit_err += 1

    acc_corr = hit_corr / n_corr if n_corr > 0 else 0.0
    acc_err  = hit_err  / n_err  if n_err  > 0 else 0.0
    denom = acc_corr + acc_err
    f1 = 2 * acc_corr * acc_err / denom if denom > 0 else 0.0
    return {
        "Acc_Corr": acc_corr,
        "Acc_Err":  acc_err,
        "F1":       f1,
        "n_corr":   n_corr,
        "n_err":    n_err,
    }


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


def print_domain_table(rows: List[Dict], overall: Dict, n_total: int) -> None:
    print(f"\n{'Domain':<25} {'F1':>8} {'Acc_Corr':>10} {'Acc_Err':>10} {'N_corr':>8} {'N_err':>7}")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['Domain']:<25} {r['F1']*100:>7.2f}%"
            f" {r['Acc_Corr']*100:>9.2f}%"
            f" {r['Acc_Err']*100:>9.2f}%"
            f" {r['n_corr']:>8} {r['n_err']:>7}"
        )
    print("-" * 70)
    print(
        f"{'Average':<25} {overall['F1']*100:>7.2f}%"
        f" {overall['Acc_Corr']*100:>9.2f}%"
        f" {overall['Acc_Err']*100:>9.2f}%"
        f" {overall['n_corr']:>8} {overall['n_err']:>7}"
        f"  (N={n_total})"
    )


# ---------------------------------------------------------------------------
# Inference driver
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
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


async def _infer_domain(
    items: List[Dict],
    out_file: str,
    model: str,
    api_key: str,
    base_url: str,
    mode: str,
    thinking_type: Optional[str],
    budget_tokens: int,
    max_tokens: int,
    concurrency: int,
    max_retries: int,
    backoff_cap: float,
    domain: str,
) -> List[Dict]:
    # Resume: skip already processed ids
    processed_ids: set = set()
    if os.path.exists(out_file):
        for r in load_jsonl(out_file):
            pid = r.get("id") or r.get("question_id")
            if pid is not None:
                processed_ids.add(pid)

    to_do = [it for it in items if (it.get("id") or it.get("question_id")) not in processed_ids]
    if not to_do:
        return load_jsonl(out_file) if os.path.exists(out_file) else []

    print(f"  [{domain}] {len(to_do)}/{len(items)} items remaining …")

    queue: asyncio.Queue = asyncio.Queue()
    for it in to_do:
        queue.put_nowait(it)

    write_lock = asyncio.Lock()
    pbar = tqdm(total=len(to_do), desc=f"  {domain}", leave=False)

    async def worker(sess: aiohttp.ClientSession):
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                res = await call_api(
                    sess, item, model, api_key, base_url,
                    mode, thinking_type, budget_tokens, max_tokens,
                    max_retries, backoff_cap,
                )
                if not res.get("api_failed"):
                    async with write_lock:
                        with open(out_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"  Worker error ({item.get('id')}): {e}")
            finally:
                queue.task_done()
                pbar.update(1)

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as sess:
        workers = [asyncio.create_task(worker(sess))
                   for _ in range(min(concurrency, len(to_do)))]
        await asyncio.gather(*workers)

    pbar.close()
    return load_jsonl(out_file) if os.path.exists(out_file) else []


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def _load_data(path: str) -> Dict[str, List]:
    """
    Load dataset from either:
      - a single JSONL file with a "domain" field on each record, OR
      - a directory whose sub-folders are domain names, each containing one JSONL.
    Returns dict: domain -> list of items (each item gets "domain" injected).
    """
    all_items: Dict[str, List] = defaultdict(list)
    if os.path.isdir(path):
        for domain in sorted(os.listdir(path)):
            domain_dir = os.path.join(path, domain)
            if not os.path.isdir(domain_dir):
                continue
            for fname in os.listdir(domain_dir):
                if not fname.endswith(".jsonl"):
                    continue
                fpath = os.path.join(domain_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            it = json.loads(line)
                            it.setdefault("domain", domain)
                            all_items[domain].append(it)
                        except json.JSONDecodeError:
                            continue
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    it = json.loads(line)
                    all_items[it["domain"]].append(it)
                except (json.JSONDecodeError, KeyError):
                    continue
    return all_items


# ---------------------------------------------------------------------------
# Subcommand: infer
# ---------------------------------------------------------------------------

def cmd_infer(args):
    # Resolve API key: --api_key > env var
    api_key = args.api_key or os.environ.get("LLM_API_KEY", "")
    if not api_key:
        sys.exit("Error: supply --api_key or set LLM_API_KEY environment variable.")

    model_key = args.model_key or args.model
    mode      = args.mode
    tag       = f"{model_key}_{mode}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading data from {args.data} …")
    all_items = _load_data(args.data)
    domains = sorted(all_items.keys())
    n_total_items = sum(len(v) for v in all_items.values())
    print(f"  {n_total_items} items across {len(domains)} domains: {domains}")
    print(f"  Model: {args.model}  |  Mode: {mode}  |  Thinking: {args.thinking_type or 'none'}")
    print("=" * 70)

    summary_rows = []
    all_records  = []

    for domain in domains:
        domain_dir = os.path.join(args.output_dir, domain, "llm_result")
        os.makedirs(domain_dir, exist_ok=True)
        out_file = os.path.join(domain_dir, f"{tag}.jsonl")

        records = asyncio.run(_infer_domain(
            items=all_items[domain],
            out_file=out_file,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            mode=mode,
            thinking_type=args.thinking_type,
            budget_tokens=args.budget_tokens,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            backoff_cap=args.backoff_cap,
            domain=domain,
        ))

        m = calculate_metrics(records)
        summary_rows.append({"Domain": domain, "N": len(records), **m})
        all_records.extend(records)

    overall = macro_average(summary_rows)
    print_domain_table(summary_rows, overall, len(all_records))

    # Save per-domain CSV
    csv_path = os.path.join(args.output_dir, f"{tag}_summary.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    # Append to aggregated summary JSONL
    agg_path = os.path.join(args.output_dir, "model_comparison_results.jsonl")
    _append_summary(agg_path, model_key, mode, args.thinking_type, all_records, overall, domains)
    print(f"Aggregated summary updated: {agg_path}")


def _append_summary(
    agg_path: str,
    model_key: str,
    mode: str,
    thinking_type: Optional[str],
    all_records: List[Dict],
    overall: Dict,
    domains: List[str],
) -> None:
    # Per-domain entries
    by_domain: Dict[str, List] = defaultdict(list)
    for r in all_records:
        by_domain[r.get("domain", "unknown")].append(r)

    new_entries = []
    for domain in domains:
        recs = by_domain.get(domain, [])
        m = calculate_metrics(recs)
        new_entries.append({
            "model_key":    model_key,
            "thinking_mode": mode,
            "thinking_type": thinking_type,
            "dataset":      domain,
            "best_f1":      round(m["F1"] * 100, 2),
            "acc_err":      round(m["Acc_Err"] * 100, 2),
            "acc_corr":     round(m["Acc_Corr"] * 100, 2),
            "n_err":        m["n_err"],
            "n_corr":       m["n_corr"],
            "time":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    # Deduplicate: remove any existing entries for the same (model, mode, dataset)
    existing = []
    if os.path.exists(agg_path):
        for e in load_jsonl(agg_path):
            key = (e.get("model_key"), e.get("thinking_mode"), e.get("dataset"))
            if not any(
                key == (ne["model_key"], ne["thinking_mode"], ne["dataset"])
                for ne in new_entries
            ):
                existing.append(e)

    with open(agg_path, "w", encoding="utf-8") as f:
        for e in existing + new_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Subcommand: eval
# ---------------------------------------------------------------------------

def cmd_eval(args):
    if args.input_file:
        records = load_jsonl(args.input_file)
    else:
        records = []
        for p in glob.glob(os.path.join(args.input_dir, "**", "*.jsonl"), recursive=True):
            records.extend(load_jsonl(p))

    if not records:
        print("No records found.")
        return

    print(f"Loaded {len(records)} records.")

    by_domain: Dict[str, List] = defaultdict(list)
    for r in records:
        by_domain[r.get("domain", "unknown")].append(r)

    rows = []
    for domain in sorted(by_domain):
        m = calculate_metrics(by_domain[domain])
        rows.append({"Domain": domain, "N": len(by_domain[domain]), **m})

    overall = macro_average(rows)
    print_domain_table(rows, overall, len(records))

    if args.output:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\nSummary saved to: {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: batch
# ---------------------------------------------------------------------------

def _load_batch_configs(args) -> List[Dict[str, Any]]:
    """Return the list of job configs, applying CLI overrides where needed."""
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            configs = json.load(f)
    elif args.preset:
        preset = args.preset.lower()
        if preset not in BUILTIN_PRESETS:
            sys.exit(f"Unknown preset '{preset}'. Available: {list(BUILTIN_PRESETS)}")
        configs = [dict(c) for c in BUILTIN_PRESETS[preset]]
    else:
        sys.exit("Provide --config or --preset.")

    # CLI api_key overrides placeholder in every config entry
    if args.api_key:
        for c in configs:
            c["api_key"] = args.api_key

    return configs


def cmd_batch(args):
    configs = _load_batch_configs(args)

    # Load dataset once
    print(f"Loading data from {args.data} …")
    all_items = _load_data(args.data)
    domains = sorted(all_items.keys())
    n_total = sum(len(v) for v in all_items.values())
    print(f"  {n_total} items  |  {len(domains)} domains  |  {len(configs)} job(s)")
    print("=" * 70)

    for job_idx, cfg in enumerate(configs, 1):
        model     = cfg["model"]
        model_key = cfg.get("model_key") or model
        api_key   = cfg.get("api_key", "")
        base_url  = cfg["base_url"]
        mode      = cfg.get("mode", "Fast")
        thinking_type = cfg.get("thinking_type")
        budget_tokens = int(cfg.get("budget_tokens", 8192))
        max_tokens    = int(cfg.get("max_tokens", 32768))
        concurrency   = int(cfg.get("concurrency", args.concurrency))
        max_retries   = int(cfg.get("max_retries", args.max_retries))
        backoff_cap   = float(cfg.get("backoff_cap", args.backoff_cap))
        tag = f"{model_key}_{mode}"

        if not api_key or api_key == "YOUR_API_KEY":
            print(f"[Job {job_idx}/{len(configs)}] SKIP {tag} — api_key not set.")
            continue

        print(f"\n[Job {job_idx}/{len(configs)}] {tag}  thinking={thinking_type or 'none'}")

        summary_rows = []
        all_records  = []

        for domain in domains:
            domain_dir = os.path.join(args.output_dir, domain, "llm_result")
            os.makedirs(domain_dir, exist_ok=True)
            out_file = os.path.join(domain_dir, f"{tag}.jsonl")

            records = asyncio.run(_infer_domain(
                items=all_items[domain],
                out_file=out_file,
                model=model,
                api_key=api_key,
                base_url=base_url,
                mode=mode,
                thinking_type=thinking_type,
                budget_tokens=budget_tokens,
                max_tokens=max_tokens,
                concurrency=concurrency,
                max_retries=max_retries,
                backoff_cap=backoff_cap,
                domain=domain,
            ))

            m = calculate_metrics(records)
            summary_rows.append({"Domain": domain, "N": len(records), **m})
            all_records.extend(records)

        overall = macro_average(summary_rows)
        print_domain_table(summary_rows, overall, len(all_records))

        csv_path = os.path.join(args.output_dir, f"{tag}_summary.csv")
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

        agg_path = os.path.join(args.output_dir, "model_comparison_results.jsonl")
        _append_summary(agg_path, model_key, mode, thinking_type,
                        all_records, overall, domains)
        print(f"  Summary → {csv_path}")

    print("\n" + "=" * 70)
    print("All batch jobs finished.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GR-Ben LLM evaluation via API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode_cmd", required=True)

    # ── infer ──────────────────────────────────────────────────────────────
    p_infer = sub.add_parser("infer", help="Call LLM API and save predictions.")
    p_infer.add_argument("--data",       required=True,
                         help="Path to GR-BEN.jsonl")
    p_infer.add_argument("--model",      required=True,
                         help="Model name sent to the API (e.g. gpt-4o)")
    p_infer.add_argument("--base_url",   required=True,
                         help="Chat completion endpoint URL")
    p_infer.add_argument("--api_key",    default=None,
                         help="API key (or set LLM_API_KEY env var)")
    p_infer.add_argument("--output_dir", required=True,
                         help="Root directory for output files")
    p_infer.add_argument("--model_key",  default=None,
                         help="Short name used in output filenames (default: --model)")
    p_infer.add_argument("--mode",       default="Fast",
                         choices=["Fast", "Slow", "Standard"],
                         help="Thinking mode (default: Fast)")
    p_infer.add_argument("--thinking_type", default=None,
                         choices=["reasoning_effort", "enable_thinking", "budget_tokens"],
                         help="API style for slow-thinking mode")
    p_infer.add_argument("--budget_tokens", type=int, default=8192,
                         help="Thinking budget tokens (budget_tokens style, default: 8192)")
    p_infer.add_argument("--max_tokens",    type=int, default=32768,
                         help="Max output tokens (default: 32768)")
    p_infer.add_argument("--concurrency",   type=int, default=30,
                         help="Max parallel API calls (default: 30)")
    p_infer.add_argument("--max_retries",   type=int, default=12,
                         help="Max retries per item (default: 12)")
    p_infer.add_argument("--backoff_cap",   type=float, default=600.0,
                         help="Max backoff in seconds (default: 600)")

    # ── batch ──────────────────────────────────────────────────────────────
    p_batch = sub.add_parser("batch", help="Run multiple model configs sequentially.")
    p_batch.add_argument("--data",       required=True,
                         help="Path to GR-BEN.jsonl")
    p_batch.add_argument("--output_dir", required=True,
                         help="Root directory for output files")
    src = p_batch.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", metavar="NAME",
                     help=f"Builtin preset name. Available: {list(BUILTIN_PRESETS)}")
    src.add_argument("--config", metavar="FILE",
                     help="Path to a JSON array of job configs")
    p_batch.add_argument("--api_key", default=None,
                         help="Override api_key for every job in the batch "
                              "(or set LLM_API_KEY env var)")
    p_batch.add_argument("--concurrency",  type=int, default=30)
    p_batch.add_argument("--max_retries",  type=int, default=12)
    p_batch.add_argument("--backoff_cap",  type=float, default=600.0)

    # ── eval ───────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Compute metrics from saved prediction files.")
    grp = p_eval.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input_file", help="Single JSONL result file")
    grp.add_argument("--input_dir",  help="Directory containing JSONL result files")
    p_eval.add_argument("--output", default=None,
                        help="Save summary CSV to this path")

    args = parser.parse_args()
    if args.mode_cmd == "infer":
        cmd_infer(args)
    elif args.mode_cmd == "batch":
        # Allow env-var fallback for api_key in batch mode
        if not args.api_key:
            args.api_key = os.environ.get("LLM_API_KEY")
        cmd_batch(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
