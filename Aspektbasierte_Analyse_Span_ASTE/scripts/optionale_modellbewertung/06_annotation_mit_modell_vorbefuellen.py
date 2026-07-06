#!/usr/bin/env python3
"""Prefill Span-ASTE annotation columns from model suggestions.

The result is a reviewable silver-label draft, not a human gold standard.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def split_triplets(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def text_in_sentence(sentence: str, text: str) -> bool:
    return text.strip().lower() in sentence.lower()


def parse_model_triplets(
    row: dict[str, str], keep_only: bool
) -> tuple[list[tuple[str, str, str]], list[str]]:
    triplets: list[tuple[str, str, str]] = []
    skipped: list[str] = []
    sentence = row.get("sentence", "")

    for raw_triplet in split_triplets(row.get("model_triplets", "")):
        parts = [part.strip() for part in raw_triplet.split("|")]
        if len(parts) < 3:
            skipped.append(raw_triplet)
            continue

        aspect, opinion, sentiment = parts[:3]
        qc_status = parts[3].lower() if len(parts) > 3 else ""
        if keep_only and qc_status and qc_status != "keep":
            skipped.append(raw_triplet)
            continue

        if not text_in_sentence(sentence, aspect) or not text_in_sentence(sentence, opinion):
            skipped.append(raw_triplet)
            continue

        triplets.append((aspect, opinion, sentiment.upper()))

    return triplets, skipped


def prefill(args: argparse.Namespace) -> Counter:
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
            if any(
                row.get(field, "").strip()
                for field in ["gold_aspect_text", "gold_opinion_text", "gold_sentiment"]
            ):
                counts["rows_with_existing_gold_left_unchanged"] += 1
                rows.append(row)
                continue

            triplets, skipped = parse_model_triplets(row, keep_only=args.keep_only)
            if triplets:
                row["gold_aspect_text"] = " ; ".join(triplet[0] for triplet in triplets)
                row["gold_opinion_text"] = " ; ".join(triplet[1] for triplet in triplets)
                row["gold_sentiment"] = " ; ".join(triplet[2] for triplet in triplets)
                row["notes"] = "MODELL_VORBEFUELLT_REVIEW_REQUIRED"
                counts["rows_prefilled"] += 1
                counts["triplets_prefilled"] += len(triplets)
            else:
                counts["rows_left_empty"] += 1

            if skipped:
                counts["model_triplets_skipped"] += len(skipped)
                existing_note = row.get("notes", "").strip()
                skip_note = "SKIPPED_MODEL_TRIPLETS: " + " ; ".join(skipped)
                row["notes"] = f"{existing_note} | {skip_note}" if existing_note else skip_note

            rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = dict(sorted(counts.items()))
    summary["label_type"] = "silver_model_prefill_review_required"
    summary["model_filter"] = "keep_only" if args.keep_only else "all_model_triplets"
    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    return Counter(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefill Span-ASTE gold columns from model suggestions."
    )
    parser.add_argument(
        "--input",
        default="annotationen/span_aste_annotation_1000.csv",
        help="Annotation CSV to prefill.",
    )
    parser.add_argument(
        "--output",
        default="annotationen/span_aste_annotation_1000_vorbefuellt.csv",
        help="Prefilled review CSV to write.",
    )
    parser.add_argument(
        "--summary",
        default="annotationen/span_aste_annotation_1000_vorbefuellt_summary.json",
        help="Summary JSON to write.",
    )
    parser.add_argument(
        "--all-model-triplets",
        action="store_false",
        dest="keep_only",
        help="Include model triplets marked review/drop instead of only keep triplets.",
    )
    parser.set_defaults(keep_only=True)
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = prefill(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
