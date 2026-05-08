"""
example_load_data.py — Demonstrates how to load and explore GR-BEN.jsonl.

Usage
-----
python evaluation/example_load_data.py --data data/GR-BEN.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict


def load_dataset(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def print_stats(samples):
    print(f"\n{'='*60}")
    print(f"Total samples: {len(samples)}")
    print(f"{'='*60}")

    # Domain distribution
    domain_counts = Counter(s["domain"] for s in samples)
    print("\nDomain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain:<25} {count}")

    # Label distribution
    n_correct = sum(1 for s in samples if s["label"] == -1)
    n_error = sum(1 for s in samples if s["label"] >= 0)
    print(f"\nLabel distribution:")
    print(f"  Correct (label=-1):  {n_correct}  ({n_correct/len(samples)*100:.1f}%)")
    print(f"  Error   (label>=0):  {n_error}   ({n_error/len(samples)*100:.1f}%)")

    # Error type distribution
    error_types = Counter(
        s["error_type"] for s in samples if s.get("error_type") is not None
    )
    if error_types:
        print("\nError type distribution (error samples only):")
        for etype, count in error_types.most_common():
            print(f"  {etype:<30} {count}")

    # Generator distribution
    gen_counts = Counter(s["generator"] for s in samples)
    print("\nGenerator distribution:")
    for gen, count in sorted(gen_counts.items(), key=lambda x: -x[1]):
        print(f"  {gen:<45} {count}")

    # Final answer correctness
    n_fa_correct = sum(1 for s in samples if s.get("final_answer_correct"))
    print(f"\nFinal answer correct: {n_fa_correct} / {len(samples)}  ({n_fa_correct/len(samples)*100:.1f}%)")

    # Step count statistics
    step_lengths = [len(s["steps"]) for s in samples]
    print(f"\nSteps per sample:")
    print(f"  Min: {min(step_lengths)}  Max: {max(step_lengths)}  "
          f"Mean: {sum(step_lengths)/len(step_lengths):.1f}")

    print(f"\n{'='*60}\n")


def demo_access(samples):
    """Show common access patterns."""
    print("=== Access Examples ===\n")

    # Filter by domain
    physics = [s for s in samples if s["domain"] == "physics"]
    print(f"Physics samples: {len(physics)}")

    # Filter error samples
    errors = [s for s in samples if s["label"] >= 0]
    print(f"Error samples: {len(errors)}")

    # Group by domain
    by_domain = defaultdict(list)
    for s in samples:
        by_domain[s["domain"]].append(s)
    print(f"Domains: {sorted(by_domain.keys())}")

    # Print one example
    ex = samples[0]
    print(f"\nExample record (id={ex['id']}):")
    print(f"  domain    : {ex['domain']}")
    print(f"  generator : {ex['generator']}")
    print(f"  label     : {ex['label']}")
    print(f"  error_type: {ex.get('error_type')}")
    print(f"  # steps   : {len(ex['steps'])}")
    print(f"  steps[0]  : {ex['steps'][0][:80]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/GR-BEN.jsonl",
                        help="Path to GR-BEN.jsonl")
    args = parser.parse_args()

    samples = load_dataset(args.data)
    print_stats(samples)
    demo_access(samples)


if __name__ == "__main__":
    main()
