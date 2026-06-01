import torch
import torch.nn as nn
import re

class WordTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        max_len=32,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = self.embedding(x) + self.pos_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        out = self.transformer(x, mask=causal_mask)

        last_token = out[:, -1, :]

        logits = self.fc(last_token)

        return logits

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.split()

class TransformerPredictor:
    def __init__(self, model_path="transformermodel"):
        # If passed "gpt2" from older code, default to transformermodel
        if model_path == "gpt2":
            model_path = "transformermodel"
            
        print(f"[TransformerPredictor] Loading model from {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        config = torch.load(f"{model_path}/config-2.pt", map_location=self.device)
        self.word2idx = config["word2idx"]
        self.idx2word = config["idx2word"]
        self.seq_len = config["SEQ_LEN"]
        
        # Read architecture from config; fall back to defaults matching
        # the original training script if keys are missing
        embed_dim = config.get("embed_dim", 64)
        num_heads = config.get("num_heads", 2)
        num_layers = config.get("num_layers", 1)
        hidden_dim = config.get("hidden_dim", 128)
        
        self.model = WordTransformer(
            vocab_size=len(self.word2idx),
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            max_len=self.seq_len,
            dropout=0.1
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(f"{model_path}/model-2.pt", map_location=self.device))
        self.model.eval()

    def predict(self, text, prefix="", top_k=5, temperature=1.0):
        words = tokenize(text)
        results = []
        seen = set()

        # --- Try the model if we have context ---
        if words:
            model_words = words[-self.seq_len:]
            ids = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in model_words]

            while len(ids) < self.seq_len:
                ids.insert(0, self.word2idx["<PAD>"])

            x = torch.tensor([ids], dtype=torch.long).to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)

            top_probs, top_ids = torch.topk(probs, min(probs.size(-1), top_k * 10))

            for prob, idx in zip(top_probs[0], top_ids[0]):
                word = self.idx2word[idx.item()]

                if word in ("<PAD>", "<UNK>"):
                    continue
                if prefix and not word.startswith(prefix.lower()):
                    continue
                if word not in seen:
                    results.append(word)
                    seen.add(word)
                if len(results) >= top_k:
                    break

        # --- Fallback: prefix match + Levenshtein from vocab ---
        if len(results) < top_k and prefix:
            import Levenshtein
            prefix_lower = prefix.lower()
            vocab = [w for w in self.word2idx if w not in ("<PAD>", "<UNK>") and w not in seen]

            # 1) Words that start with the prefix (cheap, best UX)
            prefix_matches = [w for w in vocab if w.startswith(prefix_lower)]
            prefix_matches.sort(key=len)  # shorter = more likely what they mean
            for w in prefix_matches:
                results.append(w)
                seen.add(w)
                if len(results) >= top_k:
                    break

            # 2) Still short? Rank remaining vocab by Levenshtein distance
            if len(results) < top_k:
                remaining = [w for w in vocab if w not in seen]
                scored = [(Levenshtein.distance(prefix_lower, w), w) for w in remaining]
                scored.sort(key=lambda x: (x[0], x[1]))
                for _, w in scored:
                    results.append(w)
                    seen.add(w)
                    if len(results) >= top_k:
                        break

        return results

