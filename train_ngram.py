"""
Train an n-gram language model on OpenWebText for word prediction.

Usage:
    python train_ngram.py                     # default: 1% of OpenWebText (~100M tokens)
    python train_ngram.py --data_pct 2        # use 2% (~200M tokens)
    python train_ngram.py --corpus myfile.txt # use a custom text file

No GPU required — runs in a few minutes on a laptop.
The model is saved to  models/ngram-openwebtext/model.json
"""

import argparse
import os
import time

from ngram_predictor import NgramModel


DEFAULT_DATA_PCT = 1   # 1% of OpenWebText ≈ 80K docs ≈ 100M tokens
OUTPUT_DIR = os.path.join("models", "ngram-openwebtext")
MODEL_FILE = os.path.join(OUTPUT_DIR, "model.json")


def load_openwebtext_texts(data_pct: int) -> list:
    """Load OpenWebText subset as a list of text strings."""
    from datasets import load_dataset

    pct = max(1, min(100, data_pct))
    print(f"Loading OpenWebText ({pct}% subset, ~{pct * 100}M tokens) …")
    ds = load_dataset("Skylion007/openwebtext", split=f"train[:{pct}%]")
    texts = [row["text"] for row in ds if row["text"].strip()]
    print(f"  {len(texts):,} documents loaded")
    return texts


def load_file_texts(path: str) -> list:
    """Load a plain text file as a list of lines."""
    print(f"Loading corpus from {path} …")
    with open(path, "r", encoding="utf-8") as f:
        texts = [line for line in f if line.strip()]
    print(f"  {len(texts):,} non-empty lines loaded")
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Train n-gram model for word prediction"
    )
    parser.add_argument(
        "--data_pct", type=int, default=DEFAULT_DATA_PCT,
        help=f"Percentage of OpenWebText to use (default: {DEFAULT_DATA_PCT})",
    )
    parser.add_argument(
        "--corpus", type=str, default=None,
        help="Path to a custom text file (overrides OpenWebText)",
    )
    parser.add_argument(
        "--min_count", type=int, default=2,
        help="Minimum word frequency to include in vocabulary (default: 2)",
    )
    parser.add_argument(
        "--output", type=str, default=MODEL_FILE,
        help=f"Output path for the model file (default: {MODEL_FILE})",
    )
    args = parser.parse_args()

    # Load corpus
    if args.corpus:
        texts = load_file_texts(args.corpus)
    else:
        texts = load_openwebtext_texts(args.data_pct)

    # Train
    print("\nTraining n-gram model …")
    t0 = time.time()
    model = NgramModel(min_count=args.min_count)
    model.train(texts)
    elapsed = time.time() - t0
    print(f"  Training took {elapsed:.1f}s")

    # Save
    print(f"\nSaving model …")
    model.save(args.output)

    # Quick sanity check
    print("\n--- Sanity check: top 20 unigrams ---")
    top_words = model.unigram_counts.most_common(20)
    for w, c in top_words:
        print(f"  {w:15s} {c:>8,}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
