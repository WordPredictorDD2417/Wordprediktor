# WordPrediktor

Real-time word prediction web app powered by **N-gram** and **Transformer** language models with Levenshtein-based autocorrect.

## Features

- **N-gram predictor** — Trigram language model with linear interpolation smoothing (unigram + bigram + trigram)
- **Transformer predictor** — Custom word-level Transformer encoder trained on text data
- **Autocorrect** — Levenshtein distance–based spelling correction ranked by language model probability
- **Live suggestions** — Keyboard-style suggestion bar that updates as you type

## Project Structure

```
Wordprediktor/
├── app.py                     # Flask web server
├── ngram_predictor.py         # N-gram model: training, loading, prediction
├── transformer_predictor.py   # Transformer model: architecture and prediction
├── convert_to_pkl.py          # Convert model.json → model.pkl for faster loading
├── train_ngram_colab.ipynb    # Jupyter notebook for training the N-gram model
├── templates/
│   └── index.html             # Frontend UI (Jinja2 template)
├── static/
│   └── css/
│       └── style.css          # Styles
└── models/
    ├── ngram-openwebtext/
    │   └── model.json         # Trained N-gram model
    └── transformer/
        ├── config-2.pt        # Transformer config (vocab, hyperparams)
        ├── model-2.pt         # Transformer weights
        └── checkpoint-2.pt    # Training checkpoint
```

## Requirements

- Python 3.8+
- [Flask](https://flask.palletsprojects.com/)
- [python-Levenshtein](https://pypi.org/project/python-Levenshtein/)
- [PyTorch](https://pytorch.org/) (only required for the Transformer model)

## Setup

### 1. Install dependencies

```bash
pip install flask python-Levenshtein
```

To also use the Transformer model:

```bash
pip install torch
```

### 2. Run the app

```bash
python app.py
```

The server starts at **http://127.0.0.1:5000**. Open this URL in your browser.

### 3. Use it

1. Start typing in the text area
2. Click any suggestion in the bar below to insert it
3. Toggle between **N-gram** and **Transformer** using the buttons at the top

## Training Your Own N-gram Model

Use the provided Colab notebook [`train_ngram_colab.ipynb`](train_ngram_colab.ipynb) to train on your own text data. The trained model is saved as `model.json` inside the `models/` directory.

To convert a JSON model to Pickle for faster load times:

```bash
cd models/ngram-openwebtext
python ../../convert_to_pkl.py
```

Then update the model path in `app.py` to use `model.pkl` instead of `model.json`.

## How It Works

### N-gram Model

The N-gram predictor uses trigram language modeling with **linear interpolation smoothing**:

```
P(w | context) = λ₁·P_uni(w) + λ₂·P_bi(w|w₁) + λ₃·P_tri(w|w₁,w₂)
```

Default weights: `λ₁=0.1, λ₂=0.3, λ₃=0.6`

### Transformer Model

A custom word-level Transformer encoder with causal masking. Takes the last `N` words as context and predicts the next word from the vocabulary.

### Autocorrect

For each partially typed word, candidates within Levenshtein distance ≤ `min_dist + 1` are retrieved from the vocabulary and re-ranked by N-gram probability given the preceding context.