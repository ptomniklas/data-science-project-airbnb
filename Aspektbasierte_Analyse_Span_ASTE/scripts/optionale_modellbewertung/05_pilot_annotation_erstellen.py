#!/usr/bin/env python3
"""Create a small gold-annotation pilot sheet for Airbnb Span-ASTE.

The sheet is sentence-level. Existing pretrained Span-ASTE predictions are
included only as suggestions; the gold columns are intentionally blank.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


METADATA_FIELDS = ["listing_id", "review_id", "sentence_id", "date", "sentence"]
OUTPUT_FIELDS = [
    "annotation_id",
    "aste_row_id",
    *METADATA_FIELDS,
    "model_triplets",
    "model_triplet_count",
    "model_sentiments",
    "model_aspect_categories",
    "gold_aspect_text",
    "gold_opinion_text",
    "gold_sentiment",
    "gold_aspect_start",
    "gold_aspect_end",
    "gold_opinion_start",
    "gold_opinion_end",
    "difficult",
    "notes",
]


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def load_triplets(path: Path) -> dict[str, list[dict[str, str]]]:
    triplets_by_row: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            triplets_by_row[row["aste_row_id"]].append(row)
    return triplets_by_row


def summarize_triplets(triplets: list[dict[str, str]]) -> tuple[str, str, str]:
    if not triplets:
        return "", "", ""

    suggestions = []
    sentiments = []
    categories = []
    for row in triplets:
        suggestions.append(
            "{aspect} | {opinion} | {sentiment} | {status} | {category}".format(
                aspect=row.get("aspect", ""),
                opinion=row.get("opinion", ""),
                sentiment=row.get("sentiment", ""),
                status=row.get("qc_status", ""),
                category=row.get("aspect_category", ""),
            )
        )
        if row.get("sentiment"):
            sentiments.append(row["sentiment"])
        if row.get("aspect_category"):
            categories.append(row["aspect_category"])

    return (
        " ; ".join(suggestions),
        "|".join(sorted(set(sentiments))),
        "|".join(sorted(set(categories))),
    )


def pick_rows(
    metadata_rows: list[dict[str, str]],
    triplets_by_row: dict[str, list[dict[str, str]]],
    sample_size: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    with_triplets = [row for row in metadata_rows if triplets_by_row.get(row["aste_row_id"])]
    without_triplets = [
        row for row in metadata_rows if not triplets_by_row.get(row["aste_row_id"])
    ]

    sentiment_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in with_triplets:
        sentiments = {triplet.get("sentiment", "") for triplet in triplets_by_row[row["aste_row_id"]]}
        if "NEG" in sentiments:
            sentiment_groups["NEG"].append(row)
        elif "NEU" in sentiments:
            sentiment_groups["NEU"].append(row)
        else:
            sentiment_groups["POS"].append(row)

    selected: list[dict[str, str]] = []
    target_counts = {
        "NEG": min(10, len(sentiment_groups["NEG"])),
        "NEU": min(5, len(sentiment_groups["NEU"])),
        "POS": min(25, len(sentiment_groups["POS"])),
        "NO_TRIPLET": min(10, len(without_triplets)),
    }

    for label in ["NEG", "NEU", "POS"]:
        group = sentiment_groups[label][:]
        rng.shuffle(group)
        selected.extend(group[: target_counts[label]])

    rng.shuffle(without_triplets)
    selected.extend(without_triplets[: target_counts["NO_TRIPLET"]])

    seen = {row["aste_row_id"] for row in selected}
    remaining = [row for row in metadata_rows if row["aste_row_id"] not in seen]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    selected = selected[:sample_size]
    selected.sort(key=lambda row: int(row["aste_row_id"]))
    return selected


def create_sheet(args: argparse.Namespace) -> Counter:
    metadata_rows = load_metadata(Path(args.metadata))
    triplets_by_row = load_triplets(Path(args.triplets))
    selected_rows = pick_rows(metadata_rows, triplets_by_row, args.sample_size, args.seed)

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for annotation_id, row in enumerate(selected_rows, start=1):
            triplets = triplets_by_row.get(row["aste_row_id"], [])
            model_triplets, sentiments, categories = summarize_triplets(triplets)
            if triplets:
                counts["sentences_with_model_triplets"] += 1
            else:
                counts["sentences_without_model_triplets"] += 1
            for sentiment in filter(None, sentiments.split("|")):
                counts[f"model_sentiment_{sentiment}"] += 1

            writer.writerow(
                {
                    "annotation_id": annotation_id,
                    "aste_row_id": row["aste_row_id"],
                    "listing_id": row.get("listing_id", ""),
                    "review_id": row.get("review_id", ""),
                    "sentence_id": row.get("sentence_id", ""),
                    "date": row.get("date", ""),
                    "sentence": row.get("sentence", ""),
                    "model_triplets": model_triplets,
                    "model_triplet_count": len(triplets),
                    "model_sentiments": sentiments,
                    "model_aspect_categories": categories,
                    "gold_aspect_text": "",
                    "gold_opinion_text": "",
                    "gold_sentiment": "",
                    "gold_aspect_start": "",
                    "gold_aspect_end": "",
                    "gold_opinion_start": "",
                    "gold_opinion_end": "",
                    "difficult": "",
                    "notes": "",
                }
            )
            counts["sentences_written"] += 1

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(dict(sorted(counts.items())), outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a pilot annotation CSV for Airbnb Span-ASTE gold labels."
    )
    parser.add_argument(
        "--metadata",
        default="Span-ASTE/data_airbnb/01_span_aste_eingabe_1000_saetze_metadaten.csv",
        help="Sentence metadata aligned with the Span-ASTE sample.",
    )
    parser.add_argument(
        "--triplets",
        default="Span-ASTE/data_airbnb/04_triplets_qualitaetspruefung_1000_saetze.csv",
        help="QC-enriched pretrained Span-ASTE triplets.",
    )
    parser.add_argument(
        "--output",
        default="annotationen/span_aste_pilot_annotation_50.csv",
        help="Annotation CSV to write.",
    )
    parser.add_argument(
        "--summary",
        default="annotationen/span_aste_pilot_annotation_50_summary.json",
        help="Summary JSON to write.",
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = create_sheet(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
