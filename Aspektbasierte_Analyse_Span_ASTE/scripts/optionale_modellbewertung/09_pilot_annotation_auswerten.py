#!/usr/bin/env python3
"""Evaluate Span-ASTE pilot gold labels against model suggestions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


OUTPUT_FIELDS = [
    "annotation_id",
    "aste_row_id",
    "sentence",
    "gold_triplets",
    "model_triplets",
    "missing_gold_triplets",
    "extra_model_triplets",
    "matched_triplets",
    "gold_count",
    "model_count",
    "matched_count",
    "status",
    "notes",
]


def split_gold_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_sentiment(value: str) -> str:
    return value.strip().upper()


def make_triplet(aspect: str, opinion: str, sentiment: str) -> tuple[str, str, str]:
    return (
        normalize_text(aspect),
        normalize_text(opinion),
        normalize_sentiment(sentiment),
    )


def parse_gold_triplets(row: dict[str, str]) -> list[tuple[str, str, str]]:
    aspects = split_gold_list(row.get("gold_aspect_text", ""))
    opinions = split_gold_list(row.get("gold_opinion_text", ""))
    sentiments = split_gold_list(row.get("gold_sentiment", ""))

    if not aspects and not opinions and not sentiments:
        return []

    if not (len(aspects) == len(opinions) == len(sentiments)):
        raise ValueError(
            "Gold list length mismatch in annotation_id "
            f"{row.get('annotation_id', '')}: "
            f"{len(aspects)} aspects, {len(opinions)} opinions, "
            f"{len(sentiments)} sentiments"
        )

    return [
        make_triplet(aspect, opinion, sentiment)
        for aspect, opinion, sentiment in zip(aspects, opinions, sentiments)
    ]


def parse_model_triplets(row: dict[str, str], keep_only: bool) -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    for raw_triplet in split_gold_list(row.get("model_triplets", "")):
        parts = [part.strip() for part in raw_triplet.split("|")]
        if len(parts) < 3:
            continue
        aspect, opinion, sentiment = parts[:3]
        qc_status = parts[3].strip().lower() if len(parts) > 3 else ""
        if keep_only and qc_status and qc_status != "keep":
            continue
        triplets.append(make_triplet(aspect, opinion, sentiment))
    return triplets


def triplets_to_text(triplets: list[tuple[str, str, str]]) -> str:
    return " ; ".join(" | ".join(triplet) for triplet in triplets)


def counter_intersection_size(left: Counter, right: Counter) -> int:
    return sum((left & right).values())


def precision_recall_f1(tp: int, predicted: int, gold: int) -> dict[str, float]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(args: argparse.Namespace) -> Counter:
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    sentiment_counts: Counter = Counter()
    detail_rows: list[dict[str, str]] = []

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            counts["sentences_seen"] += 1
            gold_triplets = parse_gold_triplets(row)
            model_triplets = parse_model_triplets(row, keep_only=args.keep_only)

            gold_counter = Counter(gold_triplets)
            model_counter = Counter(model_triplets)
            matched_counter = gold_counter & model_counter
            missing_counter = gold_counter - model_counter
            extra_counter = model_counter - gold_counter

            matched_count = sum(matched_counter.values())
            gold_count = sum(gold_counter.values())
            model_count = sum(model_counter.values())

            counts["gold_triplets"] += gold_count
            counts["model_triplets"] += model_count
            counts["matched_triplets"] += matched_count
            counts["missing_gold_triplets"] += sum(missing_counter.values())
            counts["extra_model_triplets"] += sum(extra_counter.values())
            if gold_count:
                counts["sentences_with_gold_triplets"] += 1
            else:
                counts["sentences_without_gold_triplets"] += 1
            if model_count:
                counts["sentences_with_model_triplets"] += 1
            else:
                counts["sentences_without_model_triplets"] += 1

            for aspect, opinion, sentiment in gold_triplets:
                sentiment_counts[f"gold_sentiment_{sentiment}"] += 1
            for aspect, opinion, sentiment in model_triplets:
                sentiment_counts[f"model_sentiment_{sentiment}"] += 1

            if matched_count == gold_count == model_count:
                status = "exact_match"
                counts["sentences_exact_match"] += 1
            elif matched_count:
                status = "partial_match"
                counts["sentences_partial_match"] += 1
            elif gold_count and model_count:
                status = "no_exact_triplet_match"
                counts["sentences_no_exact_triplet_match"] += 1
            elif gold_count and not model_count:
                status = "model_missed_all"
                counts["sentences_model_missed_all"] += 1
            elif model_count and not gold_count:
                status = "model_extra_only"
                counts["sentences_model_extra_only"] += 1
            else:
                status = "both_empty"
                counts["sentences_both_empty"] += 1

            detail_rows.append(
                {
                    "annotation_id": row.get("annotation_id", ""),
                    "aste_row_id": row.get("aste_row_id", ""),
                    "sentence": row.get("sentence", ""),
                    "gold_triplets": triplets_to_text(list(gold_counter.elements())),
                    "model_triplets": triplets_to_text(list(model_counter.elements())),
                    "missing_gold_triplets": triplets_to_text(
                        list(missing_counter.elements())
                    ),
                    "extra_model_triplets": triplets_to_text(
                        list(extra_counter.elements())
                    ),
                    "matched_triplets": triplets_to_text(
                        list(matched_counter.elements())
                    ),
                    "gold_count": str(gold_count),
                    "model_count": str(model_count),
                    "matched_count": str(matched_count),
                    "status": status,
                    "notes": row.get("notes", ""),
                }
            )

    metrics = precision_recall_f1(
        counts["matched_triplets"], counts["model_triplets"], counts["gold_triplets"]
    )
    for name, value in metrics.items():
        counts[f"exact_triplet_{name}"] = round(value, 4)

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary = dict(sorted((counts | sentiment_counts).items()))
    summary["model_filter"] = "keep_only" if args.keep_only else "all_model_triplets"
    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    return Counter(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Span-ASTE pilot gold labels against model suggestions."
    )
    parser.add_argument(
        "--input",
        default="annotationen/span_aste_pilot_annotation_50.csv",
        help="Pilot annotation CSV with model suggestions and gold labels.",
    )
    parser.add_argument(
        "--output",
        default="annotationen/span_aste_pilot_evaluation_50.csv",
        help="Sentence-level evaluation CSV to write.",
    )
    parser.add_argument(
        "--summary",
        default="annotationen/span_aste_pilot_evaluation_50_summary.json",
        help="Evaluation summary JSON to write.",
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
    result_counts = evaluate(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
