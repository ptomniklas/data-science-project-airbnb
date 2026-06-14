#!/usr/bin/env python3
"""Create a reproducible listing-capped Span-ASTE sentence sample."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DELIMITER = "#### #### ####"
METADATA_FIELDS = [
    "aste_row_id",
    "source_aste_row_id",
    "listing_id",
    "review_id",
    "sentence_id",
    "date",
    "sentence",
]


def create_sample(args: argparse.Namespace) -> dict[str, object]:
    rng = random.Random(args.seed)
    metadata_input = Path(args.metadata)
    input_output = Path(args.output)
    metadata_output = Path(args.output_metadata)
    summary_output = Path(args.summary)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with metadata_input.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            row["source_aste_row_id"] = row["aste_row_id"]
            groups[row["listing_id"]].append(row)

    capped_rows: list[dict[str, str]] = []
    for rows in groups.values():
        if len(rows) <= args.cap_per_listing:
            capped_rows.extend(rows)
        else:
            capped_rows.extend(rng.sample(rows, args.cap_per_listing))

    if len(capped_rows) < args.target:
        raise ValueError(
            f"Capped pool too small: {len(capped_rows)} rows for target {args.target}. "
            "Increase --cap-per-listing."
        )

    selected_rows = rng.sample(capped_rows, args.target)
    selected_rows.sort(key=lambda row: int(row["source_aste_row_id"]))

    input_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    listing_counts: Counter = Counter()
    with input_output.open("w", encoding="utf-8", newline="\n") as aste_file, (
        metadata_output.open("w", encoding="utf-8", newline="")
    ) as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for aste_row_id, row in enumerate(selected_rows, start=1):
            sentence = (row.get("sentence") or "").replace(DELIMITER, " ").strip()
            aste_file.write(f"{sentence}{DELIMITER}[]\n")
            writer.writerow(
                {
                    "aste_row_id": aste_row_id,
                    "source_aste_row_id": row.get("source_aste_row_id", ""),
                    "listing_id": row.get("listing_id", ""),
                    "review_id": row.get("review_id", ""),
                    "sentence_id": row.get("sentence_id", ""),
                    "date": row.get("date", ""),
                    "sentence": sentence,
                }
            )
            listing_counts[row.get("listing_id", "")] += 1

    summary = {
        "seed": args.seed,
        "target_sentences": args.target,
        "cap_per_listing_before_final_sample": args.cap_per_listing,
        "full_sentences_seen": sum(len(rows) for rows in groups.values()),
        "full_listings_seen": len(groups),
        "capped_pool_sentences": len(capped_rows),
        "sampled_sentences": len(selected_rows),
        "sampled_listings": len(listing_counts),
        "max_sentences_per_listing_in_sample": max(listing_counts.values()),
        "min_sentences_per_listing_in_sample": min(listing_counts.values()),
        "listings_with_1_sentence": sum(count == 1 for count in listing_counts.values()),
        f"listings_with_{args.cap_per_listing}_sentences": sum(
            count == args.cap_per_listing for count in listing_counts.values()
        ),
        "output": str(input_output),
        "output_metadata": str(metadata_output),
    }
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible Span-ASTE sample capped per listing."
    )
    parser.add_argument(
        "--metadata",
        default="Span-ASTE/data_airbnb/airbnb_aste_input_full_metadata.csv",
        help="Full metadata CSV aligned with the full Span-ASTE input.",
    )
    parser.add_argument(
        "--output",
        default="Span-ASTE/data_airbnb/airbnb_aste_input_50k_listing_stratified_seed0.txt",
        help="Sampled Span-ASTE input file to write.",
    )
    parser.add_argument(
        "--output-metadata",
        default="Span-ASTE/data_airbnb/airbnb_aste_input_50k_listing_stratified_seed0_metadata.csv",
        help="Metadata CSV aligned with the sampled Span-ASTE input.",
    )
    parser.add_argument(
        "--summary",
        default="Span-ASTE/data_airbnb/airbnb_aste_input_50k_listing_stratified_seed0_summary.json",
        help="Summary JSON for the sampling run.",
    )
    parser.add_argument("--target", type=int, default=50000)
    parser.add_argument("--cap-per-listing", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(create_sample(parse_args()), ensure_ascii=False, indent=2))
