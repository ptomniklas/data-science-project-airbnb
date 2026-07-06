#!/usr/bin/env python3
"""Add conservative inferred triplets for rows still empty after model prefill.

This creates reviewable silver labels. It intentionally leaves factual or
ambiguous sentences empty.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


POSITIVE_OPINIONS = [
    "excellent",
    "highly recommend",
    "recommend",
    "surprisingly large",
    "super easy",
    "large",
    "incredible",
    "quickly",
    "lovely",
    "cheap",
    "great",
    "easy",
    "nice",
    "informative",
    "enjoyed",
    "charming",
    "spacious",
    "ready",
    "enjoyable",
    "fabulous",
    "clean",
    "cosy",
    "cozy",
    "comfortable",
    "wonderful",
    "awesome",
    "punctual",
    "pleasant",
    "friendly",
    "prompt",
    "perfect",
    "best",
    "successful",
    "accommodating",
    "patient",
    "helpful",
    "better",
    "quiet",
    "homely",
    "fantastic",
    "peaceful",
    "light",
    "roomy",
    "warm",
    "loved",
    "beautifully situated",
    "central",
    "beautiful",
    "near by",
    "close",
    "walking distance",
    "everything we needed",
    "as described",
    "as expected",
]

NEGATIVE_OPINIONS = [
    "noisy",
    "audible",
    "complaint",
    "couldn't have been more different",
    "isn't enough",
    "way to early",
    "strict",
    "business",
]

NEUTRAL_OPINIONS = [
    "as expected",
    "as described",
]

ASPECTS = [
    "communication",
    "instructions",
    "apartment",
    "appartement",
    "flat",
    "place",
    "stay",
    "location",
    "neighborhood",
    "neighbourhood",
    "area",
    "breakfast",
    "restaurants",
    "bars",
    "supermarkets",
    "public transportation",
    "transportation",
    "underground station",
    "train",
    "tram",
    "market",
    "farmers market",
    "supermarket",
    "town",
    "city",
    "door",
    "building",
    "stairs",
    "host",
    "check in",
    "check out",
    "contact",
    "arrangements",
    "families",
    "couples",
    "trip",
    "home",
]

GENERIC_STAY_POSITIVE_RE = re.compile(
    r"\b(would|will|definitely|surely|love to|hope to).{0,30}\b(stay|return|go back|book)\b",
    re.IGNORECASE,
)


def contains(sentence: str, text: str) -> bool:
    return text.lower() in sentence.lower()


def first_present(sentence: str, candidates: list[str]) -> str:
    lowered = sentence.lower()
    present = [candidate for candidate in candidates if candidate.lower() in lowered]
    if not present:
        return ""
    return sorted(present, key=lambda item: lowered.find(item.lower()))[0]


def all_present(sentence: str, candidates: list[str]) -> list[str]:
    lowered = sentence.lower()
    present = [candidate for candidate in candidates if candidate.lower() in lowered]
    return sorted(set(present), key=lambda item: lowered.find(item.lower()))


def sentiment_for_opinion(opinion: str) -> str:
    if opinion in NEGATIVE_OPINIONS:
        return "NEG"
    if opinion in NEUTRAL_OPINIONS:
        return "NEU"
    return "POS"


def infer_triplets(sentence: str) -> list[tuple[str, str, str]]:
    lowered = sentence.lower()
    triplets: list[tuple[str, str, str]] = []

    aspects = all_present(sentence, ASPECTS)
    positives = all_present(sentence, POSITIVE_OPINIONS)
    negatives = all_present(sentence, NEGATIVE_OPINIONS)
    opinions = negatives + positives

    # Strong local pattern: opinion directly describes a nearby Airbnb aspect.
    for aspect in aspects:
        for opinion in opinions:
            aspect_index = lowered.find(aspect.lower())
            opinion_index = lowered.find(opinion.lower())
            if abs(aspect_index - opinion_index) <= 80:
                triplets.append((aspect, opinion, sentiment_for_opinion(opinion)))

    # Recommendation / return sentences usually evaluate the stay/place.
    if not triplets and GENERIC_STAY_POSITIVE_RE.search(sentence):
        aspect = first_present(sentence, ["apartment", "appartement", "flat", "place", "stay", "here", "there"])
        opinion = first_present(sentence, ["highly recommend", "recommend", "stay", "return", "go back", "book"])
        if aspect and opinion:
            triplets.append((aspect, opinion, "POS"))

    # Common compact review fragments.
    if not triplets:
        compact_patterns = [
            (r"\bexcellent stay\b", "stay", "excellent", "POS"),
            (r"\bpleasant stay\b", "stay", "pleasant", "POS"),
            (r"\bgreat stay\b", "stay", "great", "POS"),
            (r"\bperfect location\b", "location", "perfect", "POS"),
            (r"\bgreat location\b", "location", "great", "POS"),
            (r"\bnice neighbourhood\b", "neighbourhood", "nice", "POS"),
            (r"\bnice neighborhood\b", "neighborhood", "nice", "POS"),
            (r"\bclean and cos[yz]\b", "place", "clean", "POS"),
            (r"\bvery comfortable\b", "place", "comfortable", "POS"),
            (r"\bjust what we needed\b", "place", "just what we needed", "POS"),
            (r"\bno issues\b", "stay", "no issues", "POS"),
            (r"\bcheck in\b.{0,40}\bsuper easy\b", "check in", "super easy", "POS"),
            (r"\bcheck out\b.{0,40}\bsuper easy\b", "check out", "super easy", "POS"),
            (r"\bcommunication\b.{0,80}\bincredible\b", "communication", "incredible", "POS"),
            (r"\bflat\b.{0,80}\bbetter\b", "flat", "better", "POS"),
            (r"\bapartment\b.{0,80}\bfantastic\b", "apartment", "fantastic", "POS"),
            (r"\bapartment\b.{0,80}\bpeaceful\b", "apartment", "peaceful", "POS"),
            (r"\bapartment\b.{0,80}\broomy\b", "apartment", "roomy", "POS"),
            (r"\bcontact\b.{0,80}\bstrict\b", "contact", "strict", "NEG"),
            (r"\bcheck out\b.{0,80}\bway to early\b", "check out", "way to early", "NEG"),
        ]
        for pattern, aspect, opinion, sentiment in compact_patterns:
            if re.search(pattern, sentence, re.IGNORECASE) and contains(sentence, opinion):
                if contains(sentence, aspect):
                    triplets.append((aspect, opinion, sentiment))
                elif aspect == "place":
                    explicit_aspect = first_present(sentence, ["apartment", "appartement", "flat", "place", "stay"])
                    if explicit_aspect:
                        triplets.append((explicit_aspect, opinion, sentiment))

    # De-duplicate while preserving order.
    seen: set[tuple[str, str, str]] = set()
    unique: list[tuple[str, str, str]] = []
    for triplet in triplets:
        if triplet not in seen:
            seen.add(triplet)
            unique.append(triplet)
    return unique[:4]


def complete(args: argparse.Namespace) -> Counter:
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        rows: list[dict[str, str]] = []
        for row in reader:
            counts["rows_seen"] += 1
            if row.get("gold_aspect_text", "").strip():
                counts["rows_existing_labels_left_unchanged"] += 1
                rows.append(row)
                continue

            triplets = infer_triplets(row.get("sentence", ""))
            if not triplets:
                counts["rows_still_empty"] += 1
                rows.append(row)
                continue

            row["gold_aspect_text"] = " ; ".join(triplet[0] for triplet in triplets)
            row["gold_opinion_text"] = " ; ".join(triplet[1] for triplet in triplets)
            row["gold_sentiment"] = " ; ".join(triplet[2] for triplet in triplets)
            existing_note = row.get("notes", "").strip()
            note = "REGELBASIERT_ERGAENZT_REVIEW_REQUIRED"
            row["notes"] = f"{existing_note} | {note}" if existing_note else note
            counts["rows_completed"] += 1
            counts["triplets_completed"] += len(triplets)
            rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = dict(sorted(counts.items()))
    summary["label_type"] = "silver_model_prefill_plus_rule_based_completion_review_required"
    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    return Counter(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete empty Span-ASTE annotation rows with conservative inferred triplets."
    )
    parser.add_argument(
        "--input",
        default="annotationen/span_aste_annotation_1000_vorbefuellt.csv",
        help="Prefilled annotation CSV.",
    )
    parser.add_argument(
        "--output",
        default="annotationen/span_aste_annotation_1000_ergaenzt.csv",
        help="Completed review CSV to write.",
    )
    parser.add_argument(
        "--summary",
        default="annotationen/span_aste_annotation_1000_ergaenzt_summary.json",
        help="Summary JSON to write.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = complete(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
