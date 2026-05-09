"""
Train (fine-tune) GPT-2 small on WikiText-2 for causal language modeling.

Usage:
    python train_transformer.py                        # full training
    python train_transformer.py --epochs 1 --debug     # quick debug run on tiny subset
    python train_transformer.py --resume               # resume from last checkpoint

The trained model and tokenizer are saved to  models/gpt2-wikitext2/
"""

import argparse
import os
import math

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2"                       # 124M params, BPE tokenizer
OUTPUT_DIR = os.path.join("models", "gpt2-wikitext2")
BLOCK_SIZE = 256                          # context window for training
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8                    # per-device; adjust to your GPU RAM
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
SAVE_STEPS = 2000
LOGGING_STEPS = 200


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_and_tokenize(tokenizer, debug: bool = False):
    """Load WikiText-2 and tokenize into fixed-length blocks."""

    print("Loading WikiText-2 (raw) …")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    if debug:
        # Use a tiny slice for fast iteration
        raw["train"] = raw["train"].select(range(5000))
        raw["validation"] = raw["validation"].select(range(500))
        raw["test"] = raw["test"].select(range(500))

    # --- tokenize --------------------------------------------------------
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=False,       # we group ourselves
            return_attention_mask=False,
        )

    print("Tokenizing …")
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # --- group into fixed-length blocks ----------------------------------
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        # drop the small remainder
        total_len = (total_len // BLOCK_SIZE) * BLOCK_SIZE
        result = {
            k: [t[i : i + BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
            for k, t in concatenated.items()
        }
        # labels = input_ids (causal LM — the Trainer shifts internally)
        result["labels"] = result["input_ids"].copy()
        return result

    print(f"Grouping into blocks of {BLOCK_SIZE} tokens …")
    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        desc="Grouping",
    )

    return lm_dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on WikiText-2")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--debug", action="store_true",
                        help="Use a tiny data slice for fast iteration")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision (requires CUDA GPU)")
    args = parser.parse_args()

    # --- tokenizer -------------------------------------------------------
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT-2 has no pad token by default; use eos as pad
    tokenizer.pad_token = tokenizer.eos_token

    # --- data ------------------------------------------------------------
    lm_dataset = load_and_tokenize(tokenizer, debug=args.debug)
    print(f"Train examples : {len(lm_dataset['train']):,}")
    print(f"Val   examples : {len(lm_dataset['validation']):,}")

    # --- model -----------------------------------------------------------
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    # --- data collator ---------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM
    )

    # --- training args ---------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=args.fp16,
        report_to="none",           # no W&B / TensorBoard
        dataloader_num_workers=2,
    )

    # --- trainer ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        data_collator=data_collator,
    )

    # --- train -----------------------------------------------------------
    resume_ckpt = None
    if args.resume:
        # Find the latest checkpoint directory
        if os.path.isdir(OUTPUT_DIR):
            ckpts = [
                os.path.join(OUTPUT_DIR, d)
                for d in os.listdir(OUTPUT_DIR)
                if d.startswith("checkpoint-")
            ]
            if ckpts:
                resume_ckpt = max(ckpts, key=os.path.getmtime)
                print(f"Resuming from checkpoint: {resume_ckpt}")

    print("Starting training …")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # --- evaluate --------------------------------------------------------
    eval_result = trainer.evaluate()
    perplexity = math.exp(eval_result["eval_loss"])
    print(f"\nValidation perplexity: {perplexity:.2f}")

    # --- save ------------------------------------------------------------
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done ✓")


if __name__ == "__main__":
    main()
