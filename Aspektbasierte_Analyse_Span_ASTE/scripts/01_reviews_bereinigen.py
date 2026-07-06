#!/usr/bin/env python3
"""Prepare Airbnb reviews for sentence-level text analysis.

The script keeps only likely English reviews, splits them into sentences, and
filters sentence fragments that are too short or too noisy for downstream NLP.
It intentionally uses only the Python standard library so it runs in this
project without additional package installation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import re
from collections import Counter
from pathlib import Path


ENGLISH_STOPWORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "apartment",
    "are", "area", "as", "at", "back", "be", "beautiful", "bed", "berlin",
    "but", "by", "can", "check", "clean", "close", "comfortable", "could",
    "definitely", "easy", "everything", "excellent", "flat", "for", "from",
    "good", "great", "had", "has", "have", "he", "helpful", "her", "here",
    "him", "his", "host", "house", "i", "in", "is", "it", "location",
    "lovely", "me", "metro", "nice", "not", "of", "on", "or", "our", "out",
    "place", "really", "recommend", "room", "she", "so", "stay", "stayed",
    "super", "that", "the", "their", "there", "they", "this", "to", "tram",
    "us", "very", "was", "we", "well", "were", "with", "would", "you",
    "your",
}

FOREIGN_MARKERS = {
    "aber", "alles", "auch", "auf", "aus", "avec", "bien", "bon", "casa",
    "con", "dans", "das", "de", "del", "der", "des", "die", "du", "ein",
    "een", "eine", "el", "en", "es", "est", "et", "für", "geen", "gli", "het",
    "ich", "il",
    "im", "ist", "la", "las", "le", "les", "lo", "ma", "mit", "muy",
    "nicht", "nous", "para", "pas", "por", "que", "sehr", "très", "und",
    "une", "van", "voor", "war", "wir", "zijn", "zu",
}

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g",
    "i.e", "u.s", "u.k", "apt", "approx", "min", "max",
}

TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Za-z0-9])")
MULTI_PUNCT_RE = re.compile(r"([!?.,]){4,}")
REPEATED_CHAR_RE = re.compile(r"([A-Za-z])\1{4,}")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"(?i)<br\s*/?>", ". ", text)
    text = TAG_RE.sub(" ", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([!?])\.+", r"\1", text)
    text = re.sub(r"\?+\.", "?", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", text)
    return text.strip()


def tokenize_words(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def is_likely_english_review(text: str) -> bool:
    words = tokenize_words(text)
    if len(words) < 5:
        return False

    alpha_chars = sum(ch.isalpha() for ch in text)
    ascii_alpha_chars = sum(ch.isascii() and ch.isalpha() for ch in text)
    ascii_alpha_ratio = ascii_alpha_chars / max(alpha_chars, 1)

    english_hits = sum(word in ENGLISH_STOPWORDS for word in words)
    foreign_hits = sum(word in FOREIGN_MARKERS for word in words)
    english_ratio = english_hits / len(words)
    foreign_ratio = foreign_hits / len(words)

    return (
        ascii_alpha_ratio >= 0.72
        and english_hits >= 2
        and english_ratio >= 0.07
        and foreign_ratio <= 0.22
    )


def is_likely_english_sentence(words: list[str], text: str) -> bool:
    alpha_chars = sum(ch.isalpha() for ch in text)
    ascii_alpha_chars = sum(ch.isascii() and ch.isalpha() for ch in text)
    ascii_alpha_ratio = ascii_alpha_chars / max(alpha_chars, 1)

    english_hits = sum(word in ENGLISH_STOPWORDS for word in words)
    foreign_hits = sum(word in FOREIGN_MARKERS for word in words)
    english_ratio = english_hits / max(len(words), 1)
    foreign_ratio = foreign_hits / max(len(words), 1)

    return (
        ascii_alpha_ratio >= 0.72
        and english_hits >= 1
        and english_ratio >= 0.08
        and foreign_ratio <= 0.20
    )


def split_sentences(text: str) -> list[str]:
    if not text:
        return []

    protected = text
    replacements: dict[str, str] = {}
    for abbr in ABBREVIATIONS:
        pattern = re.compile(rf"\b{re.escape(abbr)}\.", re.IGNORECASE)
        for match in pattern.finditer(protected):
            key = match.group(0).replace(".", "<DOT>")
            replacements[key] = match.group(0)
            protected = protected.replace(match.group(0), key)

    parts = SENTENCE_BOUNDARY_RE.split(protected)
    sentences = []
    for part in parts:
        sentence = part
        for key, value in replacements.items():
            sentence = sentence.replace(key, value)
        sentence = sentence.strip(" \t\"'")
        if sentence:
            sentences.append(sentence)
    return sentences


def sentence_filter_reason(
    sentence: str, min_words: int, min_chars: int, max_chars: int
) -> str | None:
    sentence = normalize_text(sentence)
    words = tokenize_words(sentence)
    alpha_chars = sum(ch.isalpha() for ch in sentence)
    visible_chars = sum(not ch.isspace() for ch in sentence)

    if len(sentence) < min_chars:
        return "too_short_chars"
    if len(sentence) > max_chars:
        return "too_long_chars"
    if len(words) < min_words:
        return "too_few_words"
    if not is_likely_english_sentence(words, sentence):
        return "non_english_sentence"
    if URL_RE.search(sentence):
        return "contains_url"
    if "\ufffd" in sentence:
        return "encoding_noise"
    if MULTI_PUNCT_RE.search(sentence) or REPEATED_CHAR_RE.search(sentence):
        return "repeated_noise"
    if alpha_chars / max(visible_chars, 1) < 0.55:
        return "low_alpha_ratio"
    if max((len(word) for word in words), default=0) > 28:
        return "very_long_token"
    if sum(len(word) > 2 for word in words) < 2:
        return "too_little_content"
    return None


def process_reviews(args: argparse.Namespace) -> Counter:
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    fieldnames = ["listing_id", "review_id", "sentence_id", "date", "sentence"]

    with open_text(input_path) as infile, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for review_index, row in enumerate(reader, start=1):
            if args.max_reviews and review_index > args.max_reviews:
                break

            counts["reviews_seen"] += 1
            text = normalize_text(row.get("comments", ""))
            if not text:
                counts["reviews_empty"] += 1
                continue
            if not is_likely_english_review(text):
                counts["reviews_non_english_or_uncertain"] += 1
                continue

            kept_for_review = 0
            for sentence in split_sentences(text):
                reason = sentence_filter_reason(
                    sentence, args.min_words, args.min_chars, args.max_chars
                )
                if reason:
                    counts[f"sentences_filtered_{reason}"] += 1
                    continue

                kept_for_review += 1
                writer.writerow(
                    {
                        "listing_id": row.get("listing_id", ""),
                        "review_id": row.get("id", ""),
                        "sentence_id": kept_for_review,
                        "date": row.get("date", ""),
                        "sentence": normalize_text(sentence),
                    }
                )
                counts["sentences_kept"] += 1

            if kept_for_review:
                counts["reviews_kept_with_sentences"] += 1
            else:
                counts["reviews_english_but_no_sentence_kept"] += 1

    counts["reviews_filtered_total"] = (
        counts["reviews_empty"] + counts["reviews_non_english_or_uncertain"]
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(counts.items())), f, ensure_ascii=False, indent=2)
        f.write("\n")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Airbnb reviews to clean English sentence-level rows."
    )
    parser.add_argument("--input", default="../reviews.csv.gz")
    parser.add_argument(
        "--output", default="zwischenprodukte/reviews_english_sentences_clean.csv"
    )
    parser.add_argument(
        "--summary", default="zwischenprodukte/reviews_english_sentences_summary.json"
    )
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=0,
        help="Optional limit for test runs. 0 means process the full input.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = process_reviews(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
