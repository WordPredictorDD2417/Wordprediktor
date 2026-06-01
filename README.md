# Wordprediktor 📇

This is a word predictor project which predicts the next words by either filling in unfinished words, suggesting the next word or correct spelling mistakes.

## Models 🤖

### N Gram Model

**N-gram**: The n-gram model takes the n-1 preceding words. 

The n-gram probability of a word $w_i$ given the previous $n - 1$ words is:

$$ P(w_i \mid w_{i-n+1}, \ldots, w_{i-1}) = \frac{\mathrm{Count}(w_{i-n+1}, \ldots, w_{i-1}, w_i)}{\mathrm{Count}(w_{i-n+1}, \ldots, w_{i-1})} $$


Where: 

- $w_i$ is the current word
- $w_{i-n+1}, \ldots, w_{i-1}$ are the previous $n-1$ words
- $\text{Count}(w_{i-n+1}, \ldots, w_i)$ is the count of the full n-gram
- $\text{Count}(w_{i-n+1}, \ldots, w_{i-1})$ is the count of the preceding $(n-1)$-gram

In our model the first line contains:
| Variable | Description |
|---|---|
| $V$ | Vocabulary size, equal to the number of unique tokens, including punctuation |
| $N$ | Corpus size, equal to the total number of tokens |

The next lines in the model contains: 
| Variable | Description |
|---|---|
| `Word ID` | Unique integer ID assigned to the token |
| `Token Name` | The token itself |
| `Token Count` | Number of times the token appears in the corpus |

Afterwards, the model takes the bigram probability between ID of the first and second token of the bigram

```text
First Token ID | Second Token ID | Natural Log Probability
```

### Transformer Model

The transformer model is a lightweight GPT-style word predictor.

**Architecture**:

| Hyperparameter | Value |
|---|---|
| Embedding dimension | 64 |
| Attention heads | 2 |
| Transformer layers | 1 |
| Feedforward hidden dim | 128 |
| Context window (`SEQ_LEN`) | 10 tokens |
| Dropout | 0.1 |

The model combines a token embedding and a learned positional embedding, then passes the sequence through a `TransformerEncoder` with a causal (upper-triangular) mask to prevent attending to future tokens. The final token's hidden state is projected via a linear layer to produce logits over the vocabulary.

**Tokenization**:
Text is lowercased, punctuation is stripped, and the top 10,000 most frequent words form the vocabulary. Unknown words are mapped to `<UNK>`.

## Functionalities 🔧

**Unfinished Word**: Given a partial token, model will retrieve vocabulary entries that token names start with the prefix, returning top-k most likely completions. 

**Next Word**: Given a compelte sentence of words, n-gram looks at final n-1 word and return highest log probability for proceeding word. 
Transformer model will encode full context and sample and predict next token from output distribution. 

**Spelling Mistake**: Given input token (In the case of spelling mistakes) not found in vocabulary, model computes Levenshtein distance and return closest match by distance. 

## Data 📈

The project uses different datasets to train the models. 
| Dataset | Description | Number of Tokens | Size | 
|---|---|---|---|
| `OpenWebText (10k-document subset)` | WebText dataset from OpenAI to train GPT-2. For computational efficiency, only the first 10,000 documents were used | ~10–20+ million* | Subset of full dataset |


## Usage Instructions 📋

1. **Download models:**
    ````
    Models can be found here: https://zenodo.org/records/20488152
    ````

2. **Clone the repository:**
    ````
   git clone https://github.com/WordPredictorDD2417/Wordprediktor
   cd Wordprediktor
    ````

3. **Create an activate a conda environment:**
    ````
    conda create --name Wordprediktor python=3.11 
    conda activate Wordprediktor
    ````
4. **Install the required packages:**
    ````
    pip install -r requirements.txt
    ````
5. **Run the program:**
    ````
    python app.py
    ````
