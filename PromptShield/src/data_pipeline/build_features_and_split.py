"""
PromptShield - Phase 2, Step 2.5
Loads the balanced dataset, extracts dense sentence embeddings, hand-engineers
5 NLP features, and produces a stratified (by category) train/val/test split.

Reads:  data/raw/combined_balanced.csv
Writes:
  data/processed/embeddings.npy          (N x 384 float32 array)
  data/processed/features.csv            (hand-engineered features, one row per prompt)
  data/processed/full_dataset.csv        (text, category, is_attack, split, row_id)
  data/processed/train.csv / val.csv / test.csv  (row_id references into the above)
"""

import os
import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from loguru import logger

logger.add("logs/feature_engineering.log", rotation="5 MB")

INPUT_PATH = "data/raw/combined_balanced.csv"
OUTPUT_DIR = "data/processed"

# ---------------------------------------------------------------------------
# Hand-engineered feature functions
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = [
    "<|", "|>", "[INST]", "[/INST]", "###", "system:", "assistant:", "user:",
    "<system>", "</system>", "{{", "}}", "<<", ">>",
]

IMPERATIVE_VERBS = [
    "ignore", "disregard", "forget", "override", "bypass", "reveal", "print",
    "act", "pretend", "become", "respond", "answer", "comply", "obey",
    "execute", "output", "generate", "tell", "show", "give", "do",
]

ROLEPLAY_LEXICON = [
    "pretend", "roleplay", "persona", "character", "dan mode", "jailbreak",
    "developer mode", "act as", "you are now", "simulate", "stay in character",
    "immersive", "fictional", "story mode",
]


def count_special_tokens(text: str) -> int:
    lower = text.lower()
    return sum(lower.count(tok.lower()) for tok in SPECIAL_TOKENS)


def imperative_verb_density(text: str) -> float:
    """Fraction of words that are imperative/command verbs."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    imperative_count = sum(1 for w in words if w in IMPERATIVE_VERBS)
    return imperative_count / len(words)


def roleplay_lexicon_score(text: str) -> float:
    """Count of roleplay-lexicon phrase hits, normalized by text length (per 100 chars)."""
    lower = text.lower()
    hits = sum(lower.count(phrase) for phrase in ROLEPLAY_LEXICON)
    return hits / max(len(text) / 100, 1)


def shannon_entropy(text: str) -> float:
    """Standard Shannon entropy of the character distribution."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = -sum((c / length) * math.log2(c / length) for c in counts.values())
    return entropy


def char_entropy_anomaly_score(text: str, corpus_mean: float, corpus_std: float) -> float:
    """Z-score of this text's entropy relative to the corpus entropy distribution.
    Computed in two passes: first pass gathers raw entropy, second pass z-scores it."""
    ent = shannon_entropy(text)
    if corpus_std == 0:
        return 0.0
    return (ent - corpus_mean) / corpus_std


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger.info(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["row_id"] = df.index
    logger.info(f"Loaded {len(df)} rows")

    # -----------------------------------------------------------------------
    # Hand-engineered features
    # -----------------------------------------------------------------------
    logger.info("Computing hand-engineered features ...")

    df["special_token_count"] = df["text"].apply(count_special_tokens)
    df["imperative_verb_density"] = df["text"].apply(imperative_verb_density)
    df["roleplay_lexicon_score"] = df["text"].apply(roleplay_lexicon_score)

    # Entropy anomaly score requires corpus-wide mean/std first
    df["_raw_entropy"] = df["text"].apply(shannon_entropy)
    entropy_mean = df["_raw_entropy"].mean()
    entropy_std = df["_raw_entropy"].std()
    df["char_entropy_anomaly_score"] = df["_raw_entropy"].apply(
        lambda e: (e - entropy_mean) / entropy_std if entropy_std > 0 else 0.0
    )
    df.drop(columns=["_raw_entropy"], inplace=True)

    # Prompt-length z-score
    df["_char_length"] = df["text"].str.len()
    length_mean = df["_char_length"].mean()
    length_std = df["_char_length"].std()
    df["prompt_length_zscore"] = df["_char_length"].apply(
        lambda l: (l - length_mean) / length_std if length_std > 0 else 0.0
    )
    df.drop(columns=["_char_length"], inplace=True)

    feature_cols = [
        "row_id",
        "special_token_count",
        "imperative_verb_density",
        "roleplay_lexicon_score",
        "char_entropy_anomaly_score",
        "prompt_length_zscore",
    ]
    features_df = df[feature_cols].copy()
    features_path = os.path.join(OUTPUT_DIR, "features.csv")
    features_df.to_csv(features_path, index=False)
    logger.info(f"Saved hand-engineered features to {features_path}")

    # -----------------------------------------------------------------------
    # Dense sentence embeddings
    # -----------------------------------------------------------------------
    logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Encoding all prompts into embeddings (this may take a minute) ...")
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings_path = os.path.join(OUTPUT_DIR, "embeddings.npy")
    np.save(embeddings_path, embeddings.astype(np.float32))
    logger.info(f"Saved embeddings of shape {embeddings.shape} to {embeddings_path}")

    # -----------------------------------------------------------------------
    # Stratified train/val/test split BY CATEGORY (not just binary label)
    # -----------------------------------------------------------------------
    logger.info("Performing stratified split by category (70/15/15) ...")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["category"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["category"], random_state=42
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=False)
    full_df = full_df.sort_values("row_id").reset_index(drop=True)

    full_dataset_path = os.path.join(OUTPUT_DIR, "full_dataset.csv")
    full_df[["row_id", "text", "category", "is_attack", "source", "split"]].to_csv(
        full_dataset_path, index=False
    )

    train_df[["row_id", "category", "is_attack"]].to_csv(
        os.path.join(OUTPUT_DIR, "train.csv"), index=False
    )
    val_df[["row_id", "category", "is_attack"]].to_csv(
        os.path.join(OUTPUT_DIR, "val.csv"), index=False
    )
    test_df[["row_id", "category", "is_attack"]].to_csv(
        os.path.join(OUTPUT_DIR, "test.csv"), index=False
    )

    logger.info(f"Saved full dataset with split labels to {full_dataset_path}")
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\n=== FEATURE ENGINEERING + SPLIT DONE ===")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Features saved: {features_path}")
    print(f"Train: {len(train_df)} rows")
    print(f"Val:   {len(val_df)} rows")
    print(f"Test:  {len(test_df)} rows")
    print("\nTrain category distribution:")
    print(train_df["category"].value_counts())
    print("\nVal category distribution:")
    print(val_df["category"].value_counts())
    print("\nTest category distribution:")
    print(test_df["category"].value_counts())


if __name__ == "__main__":
    main()
