import os
from flask import Flask, render_template, request, jsonify
from ngram_predictor import NgramPredictor
import Levenshtein

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load N-gram model
# ---------------------------------------------------------------------------
ngram_model_path = os.path.join("models", "ngram-openwebtext", "model.json")
print(f"Loading N-gram model from {ngram_model_path}")
try:
    ngram_predictor = NgramPredictor(ngram_model_path)
    ngram_vocab = list(ngram_predictor.model.vocab)
except Exception as e:
    print(f"Failed to load NgramPredictor: {e}")
    ngram_predictor = None
    ngram_vocab = []

# ---------------------------------------------------------------------------
# Load Transformer model (lazy — only if directory exists)
# ---------------------------------------------------------------------------
transformer_predictor = None
transformer_model_dir = os.path.join("models", "transformer")
if os.path.isdir(transformer_model_dir):
    try:
        from transformer_predictor import TransformerPredictor
        print("Loading Transformer model …")
        transformer_predictor = TransformerPredictor(transformer_model_dir)
    except Exception as e:
        print(f"Failed to load TransformerPredictor: {e}")


# ---------------------------------------------------------------------------
# Levenshtein auto-correct helper
# ---------------------------------------------------------------------------
def get_levenshtein_corrections(context, word, top_k=5):
    if not word:
        return []

    vocab = ngram_vocab
    if not vocab and transformer_predictor:
        vocab = list(transformer_predictor.word2idx.keys())

    if not vocab:
        return []

    word_lower = word.lower()

    scored = [
        (Levenshtein.distance(word_lower, v), v)
        for v in vocab
        if v not in ["<PAD>", "<UNK>"]
    ]

    min_dist = min(scored, key=lambda x: x[0])[0] if scored else 0
    candidates = [v for d, v in scored if d <= min_dist + 1]

    prob_scored = []

    if ngram_predictor:
        from ngram_predictor import tokenize
        context_words = tokenize(context)
        ctx = tuple(context_words[-2:]) if len(context_words) >= 2 else tuple(context_words)

        for v in candidates:
            prob = ngram_predictor.model.log_prob(v, ctx)
            dist = Levenshtein.distance(word_lower, v)
            prob_scored.append((dist, -prob, v))
    else:
        for v in candidates:
            dist = Levenshtein.distance(word_lower, v)
            prob_scored.append((dist, 0, v))

    prob_scored.sort()
    return [v for d, p, v in prob_scored[:top_k]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    has_transformer = transformer_predictor is not None
    return render_template("index.html", has_transformer=has_transformer)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    text = data.get("text", "")
    prefix = data.get("prefix", "")
    model = data.get("model", "ngram")  # "ngram" | "transformer"

    suggestions = []

    if model == "transformer" and transformer_predictor:
        suggestions = transformer_predictor.predict(text, prefix, top_k=8)
    elif ngram_predictor:
        suggestions = ngram_predictor.predict(text, prefix, top_k=8)

    corrections = []
    if prefix:
        corrections = get_levenshtein_corrections(text, prefix, top_k=5)

    return jsonify({
        "suggestions": suggestions,
        "corrections": corrections,
        "model": model,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
