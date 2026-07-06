#!/usr/bin/env python3
"""Flatten completed Span-ASTE annotation rows into one triplet per CSV row."""

from __future__ import annotations

import argparse
import csv
import json
import random
import importlib.util
from collections import Counter
from pathlib import Path


def load_classify_aspect():
    script_path = Path(__file__).with_name("04_triplets_qualitaetspruefung.py")
    spec = importlib.util.spec_from_file_location("triplets_qualitaetspruefung", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.classify_aspect


classify_aspect = load_classify_aspect()


TRIPLET_LIST_FIELDS = [
    "gold_aspect_text",
    "gold_opinion_text",
    "gold_sentiment",
    "gold_aspect_start",
    "gold_aspect_end",
    "gold_opinion_start",
    "gold_opinion_end",
]

METADATA_FIELDS = [
    "annotation_id",
    "aste_row_id",
    "listing_id",
    "review_id",
    "sentence_id",
    "date",
    "sentence",
]

OUTPUT_FIELDS = [
    *METADATA_FIELDS,
    "triplet_id",
    "triplet_index_in_sentence",
    "aspect",
    "opinion",
    "sentiment",
    "aspect_start",
    "aspect_end",
    "opinion_start",
    "opinion_end",
    "normalized_aspect",
    "aspect_category",
    "topic_category",
    "qc_status",
    "qc_reason",
    "usable_for_analysis",
    "model_triplets",
    "model_triplet_count",
    "model_sentiments",
    "model_aspect_categories",
    "notes",
]

CLEANLINESS_TERMS = {
    "clean",
    "cleaned",
    "cleanliness",
    "dirty",
    "spotless",
    "tidy",
    "untidy",
}

TOPIC_BY_ASPECT_CATEGORY = {
    "location": "location",
    "view": "location",
    "transport": "transport",
    "mobility": "transport",
    "host_communication": "host",
    "check_in": "host",
    "booking": "host",
    "comfort": "comfort",
    "amenities": "comfort",
    "interior": "comfort",
    "outdoor": "comfort",
    "family": "comfort",
    "noise": "noise",
    "surroundings_food": "food/surroundings",
    "surroundings": "food/surroundings",
    "value": "value",
    "accommodation": "accommodation",
    "building": "building",
    "overall_experience": "overall_experience",
}


def split_list_field(value: str | None) -> list[str]:
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def row_triplet_lists(row: dict[str, str]) -> dict[str, list[str]]:
    return {field: split_list_field(row.get(field, "")) for field in TRIPLET_LIST_FIELDS}


def validate_triplet_lists(
    row: dict[str, str], values: dict[str, list[str]]
) -> tuple[bool, str]:
    lengths = {field: len(items) for field, items in values.items()}
    populated_lengths = {field: length for field, length in lengths.items() if length > 0}
    if not populated_lengths:
        return False, "empty"

    unique_lengths = set(populated_lengths.values())
    if len(unique_lengths) != 1:
        return False, (
            f"list_length_mismatch annotation_id={row.get('annotation_id', '')} "
            f"lengths={lengths}"
        )

    return True, ""


def classify_topic(aspect: str, opinion: str, aspect_category: str) -> str:
    aspect_words = set(aspect.lower().replace("-", " ").split())
    opinion_words = set(opinion.lower().replace("-", " ").split())
    if aspect_words & CLEANLINESS_TERMS or opinion_words & CLEANLINESS_TERMS:
        return "cleanliness"
    return TOPIC_BY_ASPECT_CATEGORY.get(aspect_category, "other/unknown")


def flatten_annotations(args: argparse.Namespace) -> Counter:
    input_path = Path(args.input)
    output_path = Path(args.output)
    qc_sample_path = Path(args.qc_sample_output)
    summary_path = Path(args.summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    qc_sample_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    flat_rows: list[dict[str, str]] = []
    qc_candidate_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            counts["sentences_seen"] += 1
            values = row_triplet_lists(row)
            valid, reason = validate_triplet_lists(row, values)

            if not valid:
                if reason == "empty":
                    counts["sentences_without_triplets"] += 1
                else:
                    counts["sentences_skipped_list_length_mismatch"] += 1
                    skipped_rows.append({"annotation_id": row.get("annotation_id", ""), "reason": reason})
                continue

            triplet_count = len(values["gold_aspect_text"])
            counts["sentences_with_triplets"] += 1
            counts["triplets_seen"] += triplet_count

            qc_candidate_rows.append(row)

            for index in range(triplet_count):
                aspect = values["gold_aspect_text"][index]
                opinion = values["gold_opinion_text"][index]
                normalized, category, qc_status, qc_reason = classify_aspect(aspect)
                topic_category = classify_topic(aspect, opinion, category)
                triplet_id = f"{row.get('annotation_id', '')}-{index + 1}"

                out_row = {
                    field: row.get(field, "")
                    for field in METADATA_FIELDS
                }
                out_row.update(
                    {
                        "triplet_id": triplet_id,
                        "triplet_index_in_sentence": str(index + 1),
                        "aspect": aspect,
                        "opinion": opinion,
                        "sentiment": values["gold_sentiment"][index],
                        "aspect_start": values["gold_aspect_start"][index],
                        "aspect_end": values["gold_aspect_end"][index],
                        "opinion_start": values["gold_opinion_start"][index],
                        "opinion_end": values["gold_opinion_end"][index],
                        "normalized_aspect": normalized,
                        "aspect_category": category,
                        "topic_category": topic_category,
                        "qc_status": qc_status,
                        "qc_reason": qc_reason,
                        "usable_for_analysis": "yes" if qc_status == "keep" else "no",
                        "model_triplets": row.get("model_triplets", ""),
                        "model_triplet_count": row.get("model_triplet_count", ""),
                        "model_sentiments": row.get("model_sentiments", ""),
                        "model_aspect_categories": row.get("model_aspect_categories", ""),
                        "notes": row.get("notes", ""),
                    }
                )
                flat_rows.append(out_row)
                counts[f"sentiment_{out_row['sentiment']}"] += 1
                counts[f"aspect_category_{category}"] += 1
                counts[f"topic_category_{topic_category}"] += 1
                counts[f"qc_status_{qc_status}"] += 1

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(flat_rows)

    sample_size = min(args.qc_sample_size, len(qc_candidate_rows))
    random_generator = random.Random(args.random_seed)
    qc_sample = random_generator.sample(qc_candidate_rows, sample_size)
    with qc_sample_path.open("w", encoding="utf-8", newline="") as outfile:
        fieldnames = [
            *METADATA_FIELDS,
            "gold_aspect_text",
            "gold_opinion_text",
            "gold_sentiment",
            "gold_aspect_start",
            "gold_aspect_end",
            "gold_opinion_start",
            "gold_opinion_end",
            "model_triplets",
            "model_triplet_count",
            "notes",
            "manual_check_ok",
            "manual_check_notes",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in qc_sample:
            sample_row = {field: row.get(field, "") for field in fieldnames}
            sample_row["manual_check_ok"] = ""
            sample_row["manual_check_notes"] = ""
            writer.writerow(sample_row)

    counts["triplets_written"] = len(flat_rows)
    counts["qc_sample_sentences_written"] = sample_size
    counts["skipped_problem_rows"] = len(skipped_rows)

    summary = dict(sorted(counts.items()))
    if skipped_rows:
        summary["skipped_rows"] = skipped_rows

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a triplet-level analysis CSV from completed Span-ASTE annotations."
    )
    parser.add_argument(
        "--input",
        default="annotationen/span_aste_annotation_1000_ergaenzt.csv",
        help="Completed sentence-level annotation CSV.",
    )
    parser.add_argument(
        "--output",
        default="zwischenprodukte/airbnb_aste_triplets_1000_ergaenzt_flat.csv",
        help="Triplet-level CSV to write.",
    )
    parser.add_argument(
        "--qc-sample-output",
        default="qualitaetskontrolle/annotationen_stichprobe_100.csv",
        help="Random sentence-level sample for manual annotation QC.",
    )
    parser.add_argument(
        "--summary",
        default="zwischenprodukte/airbnb_aste_triplets_1000_ergaenzt_flat_summary.json",
        help="Summary JSON to write.",
    )
    parser.add_argument(
        "--qc-sample-size",
        type=int,
        default=100,
        help="Number of labeled sentences to sample for manual QC.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible QC sampling.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = flatten_annotations(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
