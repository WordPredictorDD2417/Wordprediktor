"""
N-gram word predictor using unigram, bigram, and trigram language models
with linear interpolation smoothing.

Given context text and a partial word prefix, produces a ranked list
of whole-word completions by computing interpolated n-gram probabilities
for all vocabulary words matching the prefix.

Usage (standalone test):
    python ngram_predictor.py [model_path]

As a module:
    from ngram_predictor import NgramPredictor
    predictor = NgramPredictor("models/ngram-wikitext103/model.json")
    suggestions = predictor.predict("The quick brown", "f", top_k=5)
"""

import json
import os
import re
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z]+(?:['-][a-zA-Z]+)*")


def tokenize(text: str) -> List[str]:
    """
    Extract words from *text*, keeping internal apostrophes and hyphens.

    Examples:
        "I can't believe it!"  →  ["i", "can't", "believe", "it"]
        "state-of-the-art"     →  ["state-of-the-art"]
    """
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


# ---------------------------------------------------------------------------
# NgramModel — training, smoothing, persistence
# ---------------------------------------------------------------------------
class NgramModel:
    """
    Trigram language model with linear interpolation smoothing.

    Stores:
        unigram_counts  {w: count}
        bigram_counts   {(w1, w2): count}
        trigram_counts  {(w1, w2, w3): count}
        vocab           set of all words seen >= min_count times
    """

    def __init__(
        self,
        lambdas: Tuple[float, float, float] = (0.1, 0.3, 0.6),
        min_count: int = 2,
    ):
        # Interpolation weights: (λ_uni, λ_bi, λ_tri)
        assert abs(sum(lambdas) - 1.0) < 1e-6, "Lambdas must sum to 1"
        self.lambdas = lambdas
        self.min_count = min_count

        # Raw counts
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.trigram_counts: Counter = Counter()

        # Derived after training
        self.vocab: set = set()
        self.total_tokens: int = 0

        # Context counts for conditional probabilities
        self._bigram_context: Counter = Counter()   # count(w1)
        self._trigram_context: Counter = Counter()   # count(w1, w2)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    def train(self, texts: List[str]) -> None:
        """
        Train the model on a list of text strings.
        Each string is tokenised into words; n-gram counts are accumulated.
        """
        raw_counts: Counter = Counter()

        for text in texts:
            words = tokenize(text)
            raw_counts.update(words)

            for i, w in enumerate(words):
                self.unigram_counts[w] += 1
                self.total_tokens += 1

                if i >= 1:
                    bg = (words[i - 1], w)
                    self.bigram_counts[bg] += 1
                    self._bigram_context[words[i - 1]] += 1

                if i >= 2:
                    tg = (words[i - 2], words[i - 1], w)
                    self.trigram_counts[tg] += 1
                    self._trigram_context[(words[i - 2], words[i - 1])] += 1

        # Build vocabulary: words seen >= min_count times
        self.vocab = {w for w, c in raw_counts.items() if c >= self.min_count}

        print(f"  Vocabulary size : {len(self.vocab):,}")
        print(f"  Total tokens    : {self.total_tokens:,}")
        print(f"  Unique unigrams : {len(self.unigram_counts):,}")
        print(f"  Unique bigrams  : {len(self.bigram_counts):,}")
        print(f"  Unique trigrams : {len(self.trigram_counts):,}")

    # -----------------------------------------------------------------------
    # Probability computation
    # -----------------------------------------------------------------------
    def log_prob(self, word: str, context: Tuple[str, ...] = ()) -> float:
        """
        Compute log₁₀ P(word | context) using linear interpolation.

        context is a tuple of 0, 1, or 2 preceding words.
        """
        p = self._interpolated_prob(word, context)
        if p <= 0:
            return -99.0
        return math.log10(p)

    def _interpolated_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """P(word | context) = λ1·P_uni(word) + λ2·P_bi(word|w1) + λ3·P_tri(word|w1,w2)"""
        l1, l2, l3 = self.lambdas

        # Unigram: P(w) = count(w) / N
        p_uni = self.unigram_counts.get(word, 0) / max(self.total_tokens, 1)

        # Bigram: P(w | w1)
        p_bi = 0.0
        if len(context) >= 1:
            w1 = context[-1]
            bg_count = self.bigram_counts.get((w1, word), 0)
            ctx_count = self._bigram_context.get(w1, 0)
            if ctx_count > 0:
                p_bi = bg_count / ctx_count

        # Trigram: P(w | w1, w2)
        p_tri = 0.0
        if len(context) >= 2:
            w1, w2 = context[-2], context[-1]
            tg_count = self.trigram_counts.get((w1, w2, word), 0)
            ctx_count = self._trigram_context.get((w1, w2), 0)
            if ctx_count > 0:
                p_tri = tg_count / ctx_count

        return l1 * p_uni + l2 * p_bi + l3 * p_tri

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save model to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "lambdas": list(self.lambdas),
            "min_count": self.min_count,
            "total_tokens": self.total_tokens,
            "vocab": sorted(self.vocab),
            "unigrams": dict(self.unigram_counts),
            "bigrams": {f"{k[0]}|{k[1]}": v for k, v in self.bigram_counts.items()},
            "trigrams": {
                f"{k[0]}|{k[1]}|{k[2]}": v for k, v in self.trigram_counts.items()
            },
            "bigram_ctx": dict(self._bigram_context),
            "trigram_ctx": {
                f"{k[0]}|{k[1]}": v for k, v in self._trigram_context.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Model saved to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "NgramModel":
        """Load model from a JSON or Pickle file."""
        if path.endswith(".pkl"):
            import pickle
            with open(path, "rb") as f:
                data = pickle.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        model = cls(
            lambdas=tuple(data["lambdas"]),
            min_count=data["min_count"],
        )
        model.total_tokens = data["total_tokens"]
        model.vocab = set(data["vocab"])
        model.unigram_counts = Counter(data["unigrams"])
        model.bigram_counts = Counter(
            {tuple(k.split("|")): v for k, v in data["bigrams"].items()}
        )
        model.trigram_counts = Counter(
            {tuple(k.split("|")): v for k, v in data["trigrams"].items()}
        )
        model._bigram_context = Counter(data["bigram_ctx"])
        model._trigram_context = Counter(
            {tuple(k.split("|")): v for k, v in data["trigram_ctx"].items()}
        )
        print(f"  Loaded n-gram model: {len(model.vocab):,} words, "
              f"{model.total_tokens:,} tokens")
        return model


# ---------------------------------------------------------------------------
# NgramPredictor — prediction interface
# ---------------------------------------------------------------------------
class NgramPredictor:
    """
    Word predictor backed by an n-gram language model.

    API matches TransformerPredictor so the GUI can swap between them.
    """

    def __init__(self, model_path: str):
        print(f"[NgramPredictor] Loading model from {model_path}")
        self.model = NgramModel.load(model_path)

        # Pre-sort vocab by unigram frequency for faster prefix lookups
        self._vocab_by_freq = sorted(
            self.model.vocab,
            key=lambda w: self.model.unigram_counts.get(w, 0),
            reverse=True,
        )

    def predict(
        self,
        context: str,
        prefix: str = "",
        top_k: int = 5,
    ) -> List[str]:
        """
        Return up to *top_k* whole-word completions.

        Args:
            context: Text before the word being typed (may end with space).
            prefix:  The partial word typed so far (e.g. "br").
            top_k:   Number of suggestions.
        """
        # Extract context words for n-gram lookup
        context_words = tokenize(context)
        ctx = tuple(context_words[-2:]) if len(context_words) >= 2 else tuple(context_words)

        prefix_lower = prefix.lower()

        # Find all vocab words matching the prefix
        candidates: List[Tuple[float, str]] = []
        for word in self._vocab_by_freq:
            if prefix_lower and not word.startswith(prefix_lower):
                continue
            # Skip the prefix itself if it's already a complete word
            # (we still include it as a suggestion though)
            score = self.model.log_prob(word, ctx)
            candidates.append((score, word))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # De-duplicate (shouldn't happen, but just in case)
        seen: set = set()
        results: List[str] = []
        for score, word in candidates:
            if word not in seen:
                seen.add(word)
                results.append(word)
            if len(results) >= top_k:
                break

        return results

    def predict_from_text(self, typed_text: str, top_k: int = 5) -> List[str]:
        """
        Split typed_text into context + prefix, then predict.

        "The quick br"  →  context="The quick ", prefix="br"
        "The quick "    →  context="The quick ", prefix=""
        """
        if not typed_text:
            return []

        if typed_text.endswith(" "):
            context = typed_text
            prefix = ""
        else:
            parts = typed_text.rsplit(" ", 1)
            if len(parts) == 1:
                context = ""
                prefix = parts[0]
            else:
                context = parts[0] + " "
                prefix = parts[1]

        return self.predict(context, prefix, top_k=top_k)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/ngram-wikitext103/model.json"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Run  python train_ngram.py  first.")
        sys.exit(1)

    predictor = NgramPredictor(model_path)

    test_cases = [
        "The quick brown ",
        "The quick br",
        "Machine learning is a ",
        "I went to the ",
        "pyt",
        "Artificial intelli",
        "The president of the ",
        "In the year ",
    ]

    for text in test_cases:
        suggestions = predictor.predict_from_text(text, top_k=5)
        print(f"\n  Input: '{text}'")
        print(f"  Suggestions: {suggestions}")
