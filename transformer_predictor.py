"""
Transformer-based word predictor using a fine-tuned GPT-2 model.

Given context text and a partial word prefix, produces a ranked list
of whole-word completions by expanding GPT-2's BPE subword tokens via
beam search until a word boundary (space/newline) is found in the
decoded output.

Usage (standalone test):
    python transformer_predictor.py [model_path]

As a module:
    from transformer_predictor import TransformerPredictor
    predictor = TransformerPredictor("models/gpt2-wikitext2")
    suggestions = predictor.predict("The quick brown", "f", top_k=5)
"""

import torch
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class TransformerPredictor:
    """
    Word-completion predictor backed by a GPT-2 causal language model.

    Uses beam search over BPE tokens.  A beam is "complete" when the
    decoded text of the *newly generated* tokens contains a whitespace
    character — meaning we have crossed a word boundary.
    """

    def __init__(
        self,
        model_path: str = "gpt2",
        device: Optional[str] = None,
        max_expansion_steps: int = 10,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"[TransformerPredictor] Loading '{model_path}' on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.max_expansion_steps = max_expansion_steps
        self.eos_token_id = self.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        context: str,
        prefix: str = "",
        top_k: int = 5,
        beam_width: int = 20,
    ) -> List[str]:
        """
        Return up to *top_k* whole-word completions.

        Args:
            context: Text before the word being typed (may end with space).
            prefix:  The partial word typed so far (e.g. "br").
            top_k:   Number of suggestions.
            beam_width: Beams kept at each expansion step.
        """
        if not context and not prefix:
            return []

        # Build full input text — keep the context exactly as-is and
        # append the prefix so the model is conditioned on it.
        full_text = context
        if prefix:
            if full_text and not full_text.endswith(" "):
                full_text += " "
            full_text += prefix

        input_ids = self.tokenizer.encode(
            full_text, return_tensors="pt"
        ).to(self.device)

        completions = self._beam_search(input_ids, prefix, beam_width)

        # De-duplicate (case-insensitive) and sort by score
        seen: set = set()
        unique: list = []
        for word, score in completions:
            key = word.lower()
            if key not in seen and len(key) > 0:
                seen.add(key)
                unique.append((word, score))
        unique.sort(key=lambda x: x[1], reverse=True)

        return [w for w, _ in unique[:top_k]]

    # ------------------------------------------------------------------
    # Beam search
    # ------------------------------------------------------------------
    def _beam_search(
        self,
        input_ids: torch.Tensor,
        prefix: str,
        beam_width: int,
    ) -> List[Tuple[str, float]]:
        """
        Expand BPE tokens one step at a time.

        Each beam = (cumulative_log_prob, [appended_token_ids]).

        We decode the appended tokens after every expansion.  If the
        decoded text (after stripping a possible leading space) contains
        a whitespace character, the first "word" in that text is our
        completed suggestion.
        """
        active: List[Tuple[float, list]] = [(0.0, [])]
        completed: List[Tuple[str, float]] = []
        prefix_lower = prefix.lower()

        for _step in range(self.max_expansion_steps):
            if not active:
                break

            candidates: List[Tuple[float, list]] = []

            for log_prob, appended_ids in active:
                # Compose full token sequence
                if appended_ids:
                    ext = torch.tensor(
                        [appended_ids], device=self.device
                    )
                    full_ids = torch.cat([input_ids, ext], dim=-1)
                else:
                    full_ids = input_ids

                # Forward pass
                with torch.no_grad():
                    logits = self.model(full_ids).logits[0, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                top_vals, top_idxs = torch.topk(
                    log_probs, k=min(beam_width * 3, log_probs.size(0))
                )

                for i in range(top_vals.size(0)):
                    tid = top_idxs[i].item()
                    tlp = top_vals[i].item()
                    new_lp = log_prob + tlp
                    new_app = appended_ids + [tid]

                    # Skip EOS
                    if tid == self.eos_token_id:
                        word = self._extract_word(new_app)
                        if word and word.lower().startswith(prefix_lower):
                            completed.append((word, new_lp))
                        continue

                    # Decode everything we have generated so far
                    raw = self.tokenizer.decode(
                        new_app, skip_special_tokens=True
                    )
                    # Strip one leading space (common with GPT-2 Ġ tokens)
                    text = raw.lstrip()

                    if not text:
                        # Only whitespace generated so far — keep going
                        candidates.append((new_lp, new_app))
                        continue

                    # Check for word boundary (space / newline inside text)
                    has_boundary = (
                        " " in text or "\n" in text or "\t" in text
                    )

                    if has_boundary:
                        # The first whitespace-delimited token is the word
                        word = self._clean(text.split()[0])
                        if word and word.lower().startswith(prefix_lower):
                            completed.append((word, new_lp))
                    else:
                        # Still building a single word — keep expanding
                        partial = self._clean(text)
                        if (
                            partial
                            and partial.lower().startswith(prefix_lower)
                        ):
                            candidates.append((new_lp, new_app))

            # Prune to beam_width
            candidates.sort(key=lambda x: x[0], reverse=True)
            active = candidates[:beam_width]

        # Remaining active beams → treat as completions
        for log_prob, appended_ids in active:
            word = self._extract_word(appended_ids)
            if word and word.lower().startswith(prefix_lower):
                completed.append((word, log_prob))

        return completed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(text: str) -> str:
        """Keep only alphabetic chars, hyphens, apostrophes."""
        out: list = []
        for ch in text:
            if ch.isalpha() or ch in "-'":
                out.append(ch)
            else:
                break
        return "".join(out)

    def _extract_word(self, token_ids: list) -> str:
        """Decode token ids and return the first clean word."""
        if not token_ids:
            return ""
        raw = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        text = raw.lstrip()
        if not text:
            return ""
        first = text.split()[0] if text.split() else text
        return self._clean(first)

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------
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


# -----------------------------------------------------------------------
# Quick test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    predictor = TransformerPredictor(model_path)

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
