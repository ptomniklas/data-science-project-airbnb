#!/usr/bin/env python3
"""Export a reviewed split CSV to Span-ASTE train/dev/test text files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DELIMITER = "#### #### ####"
SPLITS = ["train", "dev", "test"]


def split_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def span_list(starts: list[str], ends: list[str], index: int) -> list[int]:
    start = int(starts[index])
    end = int(ends[index])
    return list(range(start, end + 1))


def triplets_for_row(row: dict[str, str]) -> str:
    aspects = split_list(row.get("gold_aspect_text", ""))
    opinions = split_list(row.get("gold_opinion_text", ""))
    sentiments = split_list(row.get("gold_sentiment", ""))
    aspect_starts = split_list(row.get("gold_aspect_start", ""))
    aspect_ends = split_list(row.get("gold_aspect_end", ""))
    opinion_starts = split_list(row.get("gold_opinion_start", ""))
    opinion_ends = split_list(row.get("gold_opinion_end", ""))

    if not aspects and not opinions and not sentiments:
        return "[]"

    lengths = {
        len(aspects),
        len(opinions),
        len(sentiments),
        len(aspect_starts),
        len(aspect_ends),
        len(opinion_starts),
        len(opinion_ends),
    }
    if len(lengths) != 1:
        raise ValueError(
            f"Triplet/span length mismatch in annotation_id={row.get('annotation_id', '')}: "
            f"{sorted(lengths)}"
        )

    triplets = []
    for index, sentiment in enumerate(sentiments):
        triplets.append(
            (
                span_list(aspect_starts, aspect_ends, index),
                span_list(opinion_starts, opinion_ends, index),
                sentiment,
            )
        )
    return repr(triplets)


def export(args: argparse.Namespace) -> Counter:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    handles = {
        split: (output_dir / f"{args.prefix}_{split}.txt").open(
            "w", encoding="utf-8", newline="\n"
        )
        for split in SPLITS
    }

    try:
        with input_path.open("r", encoding="utf-8", newline="") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                split = row.get("split", "")
                if split not in handles:
                    counts["rows_skipped_unknown_split"] += 1
                    continue
                handles[split].write(f"{row['sentence']}{DELIMITER}{triplets_for_row(row)}\n")
                counts["rows_exported"] += 1
                counts[f"split_{split}"] += 1
                if row.get("gold_aspect_text", "").strip():
                    counts["rows_with_triplets"] += 1
                else:
                    counts["rows_without_triplets"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    summary = dict(sorted(counts.items()))
    summary["input"] = str(input_path)
    summary["output_dir"] = str(output_dir)
    summary["files"] = {
        split: str(output_dir / f"{args.prefix}_{split}.txt") for split in SPLITS
    }
    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a reviewed Gold/Silver split CSV to Span-ASTE text files."
    )
    parser.add_argument(
        "--input",
        default="data/annotations/span_aste_gold_review_450_split_ki_geprueft.csv",
        help="Reviewed split CSV with token spans.",
    )
    parser.add_argument(
        "--output-dir",
        default="Span-ASTE/data_airbnb/gold_review_450_ki_geprueft",
        help="Output directory for train/dev/test files.",
    )
    parser.add_argument(
        "--prefix",
        default="airbnb_gold_450_ki_geprueft",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--summary",
        default="data/annotations/span_aste_gold_review_450_split_ki_geprueft_export_summary.json",
        help="Summary JSON to write.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = export(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
