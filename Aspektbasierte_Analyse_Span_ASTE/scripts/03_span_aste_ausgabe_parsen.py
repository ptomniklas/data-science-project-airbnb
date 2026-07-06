#!/usr/bin/env python3
"""Parse Span-ASTE prediction output into a review-level triplet CSV."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path


DELIMITER = "#### #### ####"
METADATA_FIELDS = ["listing_id", "review_id", "sentence_id", "date", "sentence"]


def span_text(tokens: list[str], span: list[int]) -> str:
    if len(span) == 1:
        start = end = span[0]
    else:
        start, end = span[0], span[-1]
    return " ".join(tokens[start : end + 1])


def load_metadata(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        return {int(row["aste_row_id"]): row for row in reader}


def parse_output(args: argparse.Namespace) -> Counter:
    input_path = Path(args.input)
    metadata_path = Path(args.metadata)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_by_row = load_metadata(metadata_path)
    counts: Counter = Counter()
    output_fields = [
        "aste_row_id",
        *METADATA_FIELDS,
        "aspect",
        "opinion",
        "sentiment",
        "aspect_span",
        "opinion_span",
    ]

    with input_path.open("r", encoding="utf-8") as aste_file, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fields)
        writer.writeheader()

        for aste_row_id, line in enumerate(aste_file, start=1):
            counts["sentences_seen"] += 1
            line = line.rstrip("\n")
            if DELIMITER not in line:
                counts["sentences_skipped_bad_format"] += 1
                continue

            sentence, raw_triplets = line.split(DELIMITER, 1)
            tokens = sentence.split(" ")
            triplets = ast.literal_eval(raw_triplets)
            if not triplets:
                counts["sentences_without_triplets"] += 1
                continue

            metadata = metadata_by_row.get(aste_row_id, {})
            for aspect_span, opinion_span, sentiment in triplets:
                writer.writerow(
                    {
                        "aste_row_id": aste_row_id,
                        "listing_id": metadata.get("listing_id", ""),
                        "review_id": metadata.get("review_id", ""),
                        "sentence_id": metadata.get("sentence_id", ""),
                        "date": metadata.get("date", ""),
                        "sentence": metadata.get("sentence", sentence),
                        "aspect": span_text(tokens, aspect_span),
                        "opinion": span_text(tokens, opinion_span),
                        "sentiment": sentiment,
                        "aspect_span": str(aspect_span),
                        "opinion_span": str(opinion_span),
                    }
                )
                counts["triplets_written"] += 1

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(dict(sorted(counts.items())), summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse Span-ASTE predictions into a flat triplet CSV."
    )
    parser.add_argument(
        "--input",
        default="Span-ASTE/data_airbnb/02_span_aste_modellausgabe_1000_saetze.txt",
        help="Span-ASTE prediction output file.",
    )
    parser.add_argument(
        "--metadata",
        default="Span-ASTE/data_airbnb/01_span_aste_eingabe_1000_saetze_metadaten.csv",
        help="Metadata CSV created by 02_span_aste_eingabe_erstellen.py.",
    )
    parser.add_argument(
        "--output",
        default="Span-ASTE/data_airbnb/03_triplets_modell_roh_1000_saetze.csv",
        help="Flat triplet CSV to write.",
    )
    parser.add_argument(
        "--summary",
        default="Span-ASTE/data_airbnb/03_triplets_modell_roh_1000_saetze_zusammenfassung.json",
        help="Summary JSON for the parsing run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = parse_output(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
