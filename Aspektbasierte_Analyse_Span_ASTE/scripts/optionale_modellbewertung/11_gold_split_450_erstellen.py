#!/usr/bin/env python3
"""Create a reproducible 450-sentence Gold-review split for Airbnb Span-ASTE.

The source annotation file is still silver/prefilled until a human reviewer
confirms the labels. This script therefore creates a review-ready split with
explicit status columns and Span-ASTE train/dev/test files that can be used
after manual approval.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DELIMITER = "#### #### ####"
SPLIT_SIZES = {"train": 300, "dev": 75, "test": 75}
SENTIMENT_ORDER = ["NEG", "NEU", "NO_TRIPLET", "POS"]


def split_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def primary_group(row: dict[str, str]) -> str:
    sentiments = set(split_list(row.get("gold_sentiment", "")))
    if not sentiments:
        return "NO_TRIPLET"
    if "NEG" in sentiments:
        return "NEG"
    if "NEU" in sentiments:
        return "NEU"
    return "POS"


def label_source(row: dict[str, str]) -> str:
    notes = row.get("notes", "")
    has_gold = bool(row.get("gold_aspect_text", "").strip())
    if "MODELL_VORBEFUELLT" in notes and "REGELBASIERT_ERGAENZT" in notes:
        return "mixed_model_prefill_and_rule_inferred"
    if "MODELL_VORBEFUELLT" in notes:
        return "model_prefill_review_required"
    if "REGELBASIERT_ERGAENZT" in notes:
        return "rule_inferred_review_required"
    if has_gold:
        return "existing_label_review_required"
    return "no_triplet_candidate_review_required"


def allocate_counts(total: int, buckets: dict[str, int]) -> dict[str, int]:
    bucket_total = sum(buckets.values())
    raw = {
        key: (total * size / bucket_total if bucket_total else 0)
        for key, size in buckets.items()
    }
    counts = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda key: (raw[key] - counts[key], buckets[key]), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def choose_stratum_targets(
    rows_by_group: dict[str, list[dict[str, str]]], sample_size: int
) -> dict[str, int]:
    full_counts = {group: len(rows_by_group.get(group, [])) for group in SENTIMENT_ORDER}
    rare_total = full_counts["NEG"] + full_counts["NEU"]
    remaining_sample = sample_size - rare_total
    if remaining_sample < 0:
        raise ValueError("Sample size is too small for all NEG/NEU rows.")

    remaining_available = {
        "NO_TRIPLET": full_counts["NO_TRIPLET"],
        "POS": full_counts["POS"],
    }
    targets = {
        "NEG": full_counts["NEG"],
        "NEU": full_counts["NEU"],
        **allocate_counts(remaining_sample, remaining_available),
    }
    for group, target in targets.items():
        if target > full_counts[group]:
            raise ValueError(f"Not enough rows for {group}: need {target}, have {full_counts[group]}")
    return targets


def allocate_split_targets(stratum_targets: dict[str, int]) -> dict[str, dict[str, int]]:
    split_targets: dict[str, dict[str, int]] = {split: {} for split in SPLIT_SIZES}
    for group in SENTIMENT_ORDER:
        group_alloc = allocate_counts(stratum_targets[group], SPLIT_SIZES)
        for split, count in group_alloc.items():
            split_targets[split][group] = count

    # Correct possible one-off effects so every split has the requested size.
    for split, split_size in SPLIT_SIZES.items():
        diff = split_size - sum(split_targets[split].values())
        while diff != 0:
            if diff > 0:
                donor_split = max(
                    SPLIT_SIZES,
                    key=lambda candidate: sum(split_targets[candidate].values())
                    - SPLIT_SIZES[candidate],
                )
                group = max(SENTIMENT_ORDER, key=lambda candidate: split_targets[donor_split][candidate])
                split_targets[donor_split][group] -= 1
                split_targets[split][group] += 1
                diff -= 1
            else:
                receiver_split = min(
                    SPLIT_SIZES,
                    key=lambda candidate: sum(split_targets[candidate].values())
                    - SPLIT_SIZES[candidate],
                )
                group = max(SENTIMENT_ORDER, key=lambda candidate: split_targets[split][candidate])
                split_targets[split][group] -= 1
                split_targets[receiver_split][group] += 1
                diff += 1
    return split_targets


def aste_triplets(row: dict[str, str]) -> str:
    aspects = split_list(row.get("gold_aspect_text", ""))
    opinions = split_list(row.get("gold_opinion_text", ""))
    sentiments = split_list(row.get("gold_sentiment", ""))
    aspect_starts = split_list(row.get("gold_aspect_start", ""))
    aspect_ends = split_list(row.get("gold_aspect_end", ""))
    opinion_starts = split_list(row.get("gold_opinion_start", ""))
    opinion_ends = split_list(row.get("gold_opinion_end", ""))

    if not aspects:
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
        raise ValueError(f"Triplet list length mismatch in annotation_id={row.get('annotation_id')}")

    triplets = []
    for index, sentiment in enumerate(sentiments):
        aspect_span = list(range(int(aspect_starts[index]), int(aspect_ends[index]) + 1))
        opinion_span = list(range(int(opinion_starts[index]), int(opinion_ends[index]) + 1))
        triplets.append((aspect_span, opinion_span, sentiment))
    return repr(triplets)


def enriched_row(row: dict[str, str], split: str, group: str) -> dict[str, str]:
    out = dict(row)
    out["split"] = split
    out["primary_label_group"] = group
    out["label_source"] = label_source(row)
    out["gold_review_status"] = "review_required"
    out["manual_review_ok"] = ""
    out["manual_review_notes"] = ""
    return out


def create_split(args: argparse.Namespace) -> Counter:
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    aste_dir = Path(args.aste_output_dir)

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        source_fields = reader.fieldnames or []
        rows = list(reader)

    rows_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_group[primary_group(row)].append(row)

    rng = random.Random(args.seed)
    for group_rows in rows_by_group.values():
        rng.shuffle(group_rows)

    stratum_targets = choose_stratum_targets(rows_by_group, args.sample_size)
    split_targets = allocate_split_targets(stratum_targets)

    selected: list[dict[str, str]] = []
    counters_by_group = {group: 0 for group in SENTIMENT_ORDER}
    for split in ["train", "dev", "test"]:
        for group in SENTIMENT_ORDER:
            start = counters_by_group[group]
            end = start + split_targets[split][group]
            selected.extend(
                enriched_row(row, split, group) for row in rows_by_group[group][start:end]
            )
            counters_by_group[group] = end

    selected.sort(key=lambda row: (["train", "dev", "test"].index(row["split"]), int(row["annotation_id"])))

    extra_fields = [
        "split",
        "primary_label_group",
        "label_source",
        "gold_review_status",
        "manual_review_ok",
        "manual_review_notes",
    ]
    fieldnames = [*extra_fields, *source_fields]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    aste_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    split_files = {
        split: (aste_dir / f"airbnb_gold_450_{split}.txt").open("w", encoding="utf-8", newline="\n")
        for split in SPLIT_SIZES
    }
    try:
        with output_path.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in selected:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
                split = row["split"]
                group = row["primary_label_group"]
                source = row["label_source"]
                split_files[split].write(f"{row['sentence']}{DELIMITER}{aste_triplets(row)}\n")
                counts["sentences_written"] += 1
                counts[f"split_{split}"] += 1
                counts[f"group_{group}"] += 1
                counts[f"split_{split}_{group}"] += 1
                counts[f"label_source_{source}"] += 1
                if row.get("gold_aspect_text", "").strip():
                    counts["sentences_with_triplets"] += 1
                else:
                    counts["sentences_without_triplets"] += 1
    finally:
        for file_handle in split_files.values():
            file_handle.close()

    summary = dict(sorted(counts.items()))
    summary["input"] = str(input_path)
    summary["output"] = str(output_path)
    summary["aste_output_dir"] = str(aste_dir)
    summary["gold_status"] = "review_required_before_training_or_final_evaluation"
    summary["split_files"] = {
        split: str(aste_dir / f"airbnb_gold_450_{split}.txt") for split in SPLIT_SIZES
    }
    summary["stratum_targets"] = stratum_targets

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 300/75/75 Airbnb Span-ASTE Gold-review split."
    )
    parser.add_argument(
        "--input",
        default="annotationen/span_aste_annotation_1000_ergaenzt.csv",
        help="Sentence-level annotation CSV with completed candidate labels.",
    )
    parser.add_argument(
        "--output",
        default="annotationen/span_aste_gold_review_450_split.csv",
        help="Review-ready 450 sentence split CSV.",
    )
    parser.add_argument(
        "--summary",
        default="annotationen/span_aste_gold_review_450_split_summary.json",
        help="Summary JSON to write.",
    )
    parser.add_argument(
        "--aste-output-dir",
        default="Span-ASTE/data_airbnb/gold_review_450",
        help="Directory for Span-ASTE train/dev/test files.",
    )
    parser.add_argument("--sample-size", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = create_split(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
