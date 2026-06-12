#!/usr/bin/env python3
"""Convert cleaned Airbnb review sentences into Span-ASTE prediction input.

Span-ASTE expects one sentence per line in the form:

    sentence#### #### ####[]

The companion metadata CSV keeps the original Airbnb identifiers so model
predictions can be joined back to listings and reviews after inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


DELIMITER = "#### #### ####"
REQUIRED_COLUMNS = ["listing_id", "review_id", "sentence_id", "date", "sentence"]
WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_aste(sentence: str) -> str:
    sentence = WHITESPACE_RE.sub(" ", sentence or "").strip()
    return sentence.replace(DELIMITER, " ")


def convert(args: argparse.Namespace) -> Counter:
    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata)
    summary_path = Path(args.summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    metadata_fields = ["aste_row_id", *REQUIRED_COLUMNS]

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])
        ]
        if missing_columns:
            raise ValueError(f"Input file is missing columns: {missing_columns}")

        with output_path.open("w", encoding="utf-8", newline="\n") as aste_file, (
            metadata_path.open("w", encoding="utf-8", newline="")
        ) as metadata_file:
            metadata_writer = csv.DictWriter(metadata_file, fieldnames=metadata_fields)
            metadata_writer.writeheader()

            for source_row_id, row in enumerate(reader, start=1):
                if args.limit and counts["rows_written"] >= args.limit:
                    break

                counts["rows_seen"] += 1
                sentence = normalize_for_aste(row.get("sentence", ""))
                if not sentence:
                    counts["rows_skipped_empty_sentence"] += 1
                    continue

                aste_row_id = counts["rows_written"] + 1
                aste_file.write(f"{sentence}{DELIMITER}[]\n")
                metadata_writer.writerow(
                    {
                        "aste_row_id": aste_row_id,
                        "listing_id": row.get("listing_id", ""),
                        "review_id": row.get("review_id", ""),
                        "sentence_id": row.get("sentence_id", ""),
                        "date": row.get("date", ""),
                        "sentence": sentence,
                    }
                )
                counts["rows_written"] += 1
                counts["last_source_row_id"] = source_row_id

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(dict(sorted(counts.items())), summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Span-ASTE prediction input from cleaned Airbnb sentences."
    )
    parser.add_argument(
        "--input",
        default="data/processed/reviews_english_sentences_clean.csv",
        help="Clean sentence-level review CSV.",
    )
    parser.add_argument(
        "--output",
        default="Span-ASTE/data_airbnb/01_span_aste_eingabe_1000_saetze.txt",
        help="Span-ASTE input file to write.",
    )
    parser.add_argument(
        "--metadata",
        default="Span-ASTE/data_airbnb/01_span_aste_eingabe_1000_saetze_metadaten.csv",
        help="Metadata CSV aligned with the Span-ASTE input rows.",
    )
    parser.add_argument(
        "--summary",
        default="Span-ASTE/data_airbnb/01_span_aste_eingabe_1000_saetze_zusammenfassung.json",
        help="Summary JSON for the conversion run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of sentences to write. Use 0 for the full dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = convert(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
