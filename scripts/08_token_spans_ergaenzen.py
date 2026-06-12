#!/usr/bin/env python3
"""Fill token-span columns for a Span-ASTE gold annotation CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


GOLD_LIST_FIELDS = ["gold_aspect_text", "gold_opinion_text", "gold_sentiment"]
SPAN_FIELDS = [
    "gold_aspect_start",
    "gold_aspect_end",
    "gold_opinion_start",
    "gold_opinion_end",
]
TOKEN_RE = re.compile(r"\S+")


def split_gold_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def token_offsets(sentence: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in TOKEN_RE.finditer(sentence)]


def find_text_span(sentence: str, text: str) -> tuple[int, int]:
    start = sentence.lower().find(text.lower())
    if start < 0:
        raise ValueError(f"Could not find {text!r} in sentence {sentence!r}")
    return start, start + len(text)


def char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    matching = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start < char_end and token_end > char_start
    ]
    if not matching:
        raise ValueError(f"No tokens overlap char span {char_start}:{char_end}")
    return matching[0], matching[-1]


def text_to_token_span(sentence: str, text: str) -> tuple[int, int]:
    char_start, char_end = find_text_span(sentence, text)
    return char_span_to_token_span(token_offsets(sentence), char_start, char_end)


def fill_spans(args: argparse.Namespace) -> Counter:
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        for field in SPAN_FIELDS:
            if field not in fieldnames:
                fieldnames.append(field)

        rows: list[dict[str, str]] = []
        for row in reader:
            counts["rows_seen"] += 1
            aspects = split_gold_list(row.get("gold_aspect_text", ""))
            opinions = split_gold_list(row.get("gold_opinion_text", ""))
            sentiments = split_gold_list(row.get("gold_sentiment", ""))

            if not aspects and not opinions and not sentiments:
                for field in SPAN_FIELDS:
                    row[field] = ""
                counts["rows_without_gold_triplets"] += 1
                rows.append(row)
                continue

            if not (len(aspects) == len(opinions) == len(sentiments)):
                raise ValueError(
                    "Gold list length mismatch in annotation_id "
                    f"{row.get('annotation_id', '')}: "
                    f"{len(aspects)} aspects, {len(opinions)} opinions, "
                    f"{len(sentiments)} sentiments"
                )

            aspect_starts: list[str] = []
            aspect_ends: list[str] = []
            opinion_starts: list[str] = []
            opinion_ends: list[str] = []

            for aspect, opinion in zip(aspects, opinions):
                aspect_start, aspect_end = text_to_token_span(row["sentence"], aspect)
                opinion_start, opinion_end = text_to_token_span(row["sentence"], opinion)
                aspect_starts.append(str(aspect_start))
                aspect_ends.append(str(aspect_end))
                opinion_starts.append(str(opinion_start))
                opinion_ends.append(str(opinion_end))
                counts["triplet_spans_written"] += 1

            row["gold_aspect_start"] = " ; ".join(aspect_starts)
            row["gold_aspect_end"] = " ; ".join(aspect_ends)
            row["gold_opinion_start"] = " ; ".join(opinion_starts)
            row["gold_opinion_end"] = " ; ".join(opinion_ends)
            counts["rows_with_gold_triplets"] += 1
            rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(dict(sorted(counts.items())), outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add token start/end columns to a Span-ASTE gold annotation CSV."
    )
    parser.add_argument(
        "--input",
        default="data/annotations/span_aste_pilot_annotation_50.csv",
        help="Gold annotation CSV with text spans.",
    )
    parser.add_argument(
        "--output",
        default="data/annotations/span_aste_pilot_annotation_50.csv",
        help="CSV to write. Defaults to updating the input file.",
    )
    parser.add_argument(
        "--summary",
        default="data/annotations/span_aste_pilot_annotation_50_spans_summary.json",
        help="Summary JSON to write.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = fill_spans(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
