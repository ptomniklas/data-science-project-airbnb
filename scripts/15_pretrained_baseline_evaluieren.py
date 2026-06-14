#!/usr/bin/env python3
"""Evaluate Span-ASTE prediction files against a Span-ASTE gold/silver file.

The evaluation is intentionally exact-match based: aspect span, opinion span
and sentiment all have to match. This makes the reported baseline transparent
for the KI-/Silver-Annotation split and avoids hidden fuzzy matching rules.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path


DELIMITER = "#### #### ####"
OUTPUT_FIELDS = [
    "row_id",
    "sentence",
    "gold_triplets",
    "pred_triplets",
    "matched_triplets",
    "missing_gold_triplets",
    "extra_pred_triplets",
    "gold_count",
    "pred_count",
    "matched_count",
    "status",
]


def parse_line(line: str) -> tuple[str, Counter[tuple[tuple[int, ...], tuple[int, ...], str]]]:
    line = line.rstrip("\n")
    if DELIMITER not in line:
        raise ValueError(f"Bad Span-ASTE line without delimiter: {line[:120]}")
    sentence, raw_triplets = line.split(DELIMITER, 1)
    parsed = ast.literal_eval(raw_triplets.strip())
    triplets = Counter(
        (
            tuple(aspect_span),
            tuple(opinion_span),
            str(sentiment).upper(),
        )
        for aspect_span, opinion_span, sentiment in parsed
    )
    return sentence, triplets


def triplet_text(triplet: tuple[tuple[int, ...], tuple[int, ...], str]) -> str:
    aspect_span, opinion_span, sentiment = triplet
    return f"{list(aspect_span)} | {list(opinion_span)} | {sentiment}"


def counter_to_text(counter: Counter[tuple[tuple[int, ...], tuple[int, ...], str]]) -> str:
    return " ; ".join(triplet_text(triplet) for triplet in counter.elements())


def metrics(tp: int, predicted: int, gold: int) -> dict[str, float]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate(args: argparse.Namespace) -> Counter:
    gold_path = Path(args.gold)
    pred_path = Path(args.pred)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    detail_rows: list[dict[str, str]] = []

    with gold_path.open("r", encoding="utf-8") as gold_file, pred_path.open(
        "r", encoding="utf-8"
    ) as pred_file:
        for row_id, (gold_line, pred_line) in enumerate(zip(gold_file, pred_file), start=1):
            counts["sentences_seen"] += 1
            gold_sentence, gold_triplets = parse_line(gold_line)
            pred_sentence, pred_triplets = parse_line(pred_line)
            if gold_sentence != pred_sentence:
                counts["sentence_text_mismatch"] += 1

            matched = gold_triplets & pred_triplets
            missing = gold_triplets - pred_triplets
            extra = pred_triplets - gold_triplets
            gold_count = sum(gold_triplets.values())
            pred_count = sum(pred_triplets.values())
            matched_count = sum(matched.values())

            counts["gold_triplets"] += gold_count
            counts["pred_triplets"] += pred_count
            counts["matched_triplets"] += matched_count
            counts["missing_gold_triplets"] += sum(missing.values())
            counts["extra_pred_triplets"] += sum(extra.values())
            counts[f"gold_sentence_{'with' if gold_count else 'without'}_triplets"] += 1
            counts[f"pred_sentence_{'with' if pred_count else 'without'}_triplets"] += 1

            if matched_count == gold_count == pred_count:
                status = "exact_match"
            elif matched_count:
                status = "partial_match"
            elif gold_count and pred_count:
                status = "no_exact_triplet_match"
            elif gold_count:
                status = "model_missed_all"
            elif pred_count:
                status = "model_extra_only"
            else:
                status = "both_empty"
            counts[f"sentences_{status}"] += 1

            detail_rows.append(
                {
                    "row_id": str(row_id),
                    "sentence": gold_sentence,
                    "gold_triplets": counter_to_text(gold_triplets),
                    "pred_triplets": counter_to_text(pred_triplets),
                    "matched_triplets": counter_to_text(matched),
                    "missing_gold_triplets": counter_to_text(missing),
                    "extra_pred_triplets": counter_to_text(extra),
                    "gold_count": str(gold_count),
                    "pred_count": str(pred_count),
                    "matched_count": str(matched_count),
                    "status": status,
                }
            )

        remaining_gold = sum(1 for _ in gold_file)
        remaining_pred = sum(1 for _ in pred_file)
        counts["remaining_gold_lines_after_zip"] = remaining_gold
        counts["remaining_pred_lines_after_zip"] = remaining_pred

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary = dict(sorted(counts.items()))
    summary.update(metrics(counts["matched_triplets"], counts["pred_triplets"], counts["gold_triplets"]))
    summary["dataset_status"] = args.dataset_status
    summary["gold"] = str(gold_path)
    summary["pred"] = str(pred_path)
    summary["output"] = str(output_path)

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    return Counter(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exact Span-ASTE triplets in a prediction file."
    )
    parser.add_argument(
        "--gold",
        default=(
            "Span-ASTE/data_airbnb/gold_review_450_ki_geprueft/"
            "airbnb_gold_450_ki_geprueft_test.txt"
        ),
        help="Span-ASTE gold/silver test file.",
    )
    parser.add_argument(
        "--pred",
        default=(
            "Span-ASTE/data_airbnb/gold_review_450_ki_geprueft/"
            "pretrained_14res_test_pred.txt"
        ),
        help="Span-ASTE prediction file.",
    )
    parser.add_argument(
        "--output",
        default=(
            "data/annotations/pretrained_14res_ki_geprueft_test_evaluation.csv"
        ),
        help="Sentence-level evaluation CSV.",
    )
    parser.add_argument(
        "--summary",
        default=(
            "data/annotations/pretrained_14res_ki_geprueft_test_evaluation_summary.json"
        ),
        help="Evaluation summary JSON.",
    )
    parser.add_argument(
        "--dataset-status",
        default="ki_gepruefter_silver_weak_gold_split_human_check_required",
        help="Transparency note written into the summary.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = evaluate(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
