"""
PromptShield - Phase 2, Step 2.3
Fetches open prompt-injection / jailbreak datasets from Hugging Face,
tags each example into one of our 5 target categories, and saves a
combined raw CSV to data/raw/combined_raw.csv

Categories:
  - benign
  - direct_injection
  - roleplay
  - obfuscation
  - instruction_override
"""

import os
import re
import unicodedata
import pandas as pd
from datasets import load_dataset
from loguru import logger

logger.add("logs/data_fetch.log", rotation="5 MB")

# ---------------------------------------------------------------------------
# Keyword banks used for rule-based category tagging
# ---------------------------------------------------------------------------
ROLEPLAY_KEYWORDS = [
    "pretend you are", "act as", "you are now", "roleplay", "stay in character",
    "dan mode", "jailbreak", "developer mode", "do anything now", "persona",
    "simulate being", "you are dan", "ignore your previous persona",
]

INSTRUCTION_OVERRIDE_KEYWORDS = [
    "ignore previous instructions", "ignore all previous", "disregard the above",
    "forget everything", "new instructions", "override", "system prompt",
    "ignore the rules", "your new task", "from now on you will",
]

DIRECT_INJECTION_KEYWORDS = [
    "reveal your system prompt", "print your instructions", "what are your rules",
    "bypass", "unlock", "no restrictions", "without any filter",
    "respond without", "do not refuse",
]


def detect_obfuscation(text: str) -> bool:
    """Flags text using unicode tricks, excessive symbols, or base64-like patterns."""
    non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
    has_base64_pattern = bool(re.search(r"[A-Za-z0-9+/]{30,}={0,2}", text))
    has_leetspeak = bool(re.search(r"[0-9]{1}[a-zA-Z]{1}[0-9]{1}", text))
    weird_spacing = bool(re.search(r"(\w\s){10,}", text))
    return non_ascii_ratio > 0.15 or has_base64_pattern or has_leetspeak or weird_spacing


def categorize(text: str, is_attack: bool) -> str:
    """Rule-based tagging into one of the 5 target categories."""
    if not is_attack:
        return "benign"

    lower_text = text.lower()
    normalized = unicodedata.normalize("NFKD", text)

    if detect_obfuscation(normalized):
        return "obfuscation"
    if any(kw in lower_text for kw in ROLEPLAY_KEYWORDS):
        return "roleplay"
    if any(kw in lower_text for kw in INSTRUCTION_OVERRIDE_KEYWORDS):
        return "instruction_override"
    if any(kw in lower_text for kw in DIRECT_INJECTION_KEYWORDS):
        return "direct_injection"
    # Fallback bucket for attacks that don't match a specific pattern
    return "direct_injection"


def fetch_prompt_injection_dataset() -> pd.DataFrame:
    """Loads deepset/prompt-injections (binary labeled: 0=benign, 1=injection)."""
    logger.info("Loading deepset/prompt-injections ...")
    ds = load_dataset("deepset/prompt-injections")
    rows = []
    for split in ds.keys():
        for row in ds[split]:
            text = str(row["text"]).strip()
            is_attack = int(row["label"]) == 1
            if text:
                rows.append({"text": text, "is_attack": is_attack, "source": "deepset"})
    df = pd.DataFrame(rows)
    logger.info(f"deepset/prompt-injections: {len(df)} rows loaded")
    return df


def fetch_jailbreak_dataset() -> pd.DataFrame:
    """Loads rubend18/ChatGPT-Jailbreak-Prompts (all rows treated as attacks)."""
    logger.info("Loading rubend18/ChatGPT-Jailbreak-Prompts ...")
    ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts")
    rows = []
    for split in ds.keys():
        for row in ds[split]:
            text = str(row.get("Prompt", "")).strip()
            if text:
                rows.append({"text": text, "is_attack": True, "source": "rubend18_jailbreak"})
    df = pd.DataFrame(rows)
    logger.info(f"rubend18/ChatGPT-Jailbreak-Prompts: {len(df)} rows loaded")
    return df


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    df_injection = fetch_prompt_injection_dataset()
    df_jailbreak = fetch_jailbreak_dataset()

    combined = pd.concat([df_injection, df_jailbreak], ignore_index=True)
    combined.drop_duplicates(subset="text", inplace=True)
    combined = combined[combined["text"].str.len() > 3].reset_index(drop=True)

    logger.info("Applying rule-based category tagging ...")
    combined["category"] = combined.apply(
        lambda r: categorize(r["text"], r["is_attack"]), axis=1
    )

    output_path = "data/raw/combined_raw.csv"
    combined.to_csv(output_path, index=False)

    logger.info(f"Saved {len(combined)} total rows to {output_path}")
    logger.info("Category distribution:")
    logger.info(combined["category"].value_counts().to_string())

    print("\n=== DONE ===")
    print(f"Total rows: {len(combined)}")
    print(combined["category"].value_counts())
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
