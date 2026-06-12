#!/usr/bin/env python3
"""Apply a transparent AI review pass to the 450-sentence ASTE split.

This is not a human gold annotation. The script creates a separate reviewed
copy, applies only conservative corrections, and adds AI-review columns so the
result can be checked manually afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


AI_COLUMNS = ["ai_review_status", "ai_review_action", "ai_review_notes"]
SPAN_COLUMNS = [
    "gold_aspect_start",
    "gold_aspect_end",
    "gold_opinion_start",
    "gold_opinion_end",
]


HIGH_CONFIDENCE_CORRECTIONS = {
    "195": ("street", "beautiful", "POS", "no_triplet_to_positive_street"),
    "346": (
        "appartment ; appartment",
        "charming ; well located",
        "POS ; POS",
        "no_triplet_to_positive_apartment",
    ),
    "420": (
        "walking distance",
        "not really",
        "NEG",
        "no_triplet_to_negative_location_distance",
    ),
    "523": ("machinery", "loud", "NEG", "no_triplet_to_negative_noise"),
    "547": (
        "price ; listing ; elements",
        "high ; not as described ; need renovating",
        "NEG ; NEG ; NEG",
        "no_triplet_to_negative_value_listing_condition",
    ),
    "560": ("Washing machine", "worked well", "POS", "no_triplet_to_positive_amenity"),
    "642": (
        "hair drier ; everything",
        "no ; great",
        "NEG ; POS",
        "no_triplet_to_mixed_amenity_overall",
    ),
    "749": (
        "light ; light",
        "very nice ; not so nice",
        "POS ; NEG",
        "no_triplet_to_mixed_light",
    ),
    "817": (
        "apartment ; U-bahn stops",
        "very convenient ; near",
        "POS ; POS",
        "no_triplet_to_positive_location",
    ),
    "882": (
        "Everything ; loft",
        "clean ; great",
        "POS ; POS",
        "no_triplet_to_positive_cleanliness_overall",
    ),
    "348": (
        "expérience ; expérience",
        "Good ; recommand",
        "POS ; POS",
        "no_triplet_to_positive_experience",
    ),
    "461": (
        "house ; heated ; clean",
        "problems ; wasn't well ; not very clean",
        "NEG ; NEG ; NEG",
        "no_triplet_to_negative_house_condition",
    ),
    "815": (
        "amenities",
        "close",
        "POS",
        "no_triplet_to_positive_amenities_location",
    ),
    "890": ("view", "Beautiful", "POS", "no_triplet_to_positive_view"),
    "521": ("apartment", "Really liked", "POS", "no_triplet_to_positive_apartment"),
    "569": ("view", "lovely", "POS", "no_triplet_to_positive_view"),
    "953": ("place", "unique", "POS", "no_triplet_to_positive_place"),
}


HIGH_CONFIDENCE_REVISIONS = {
    "729": (
        "district",
        "best",
        "POS",
        "corrected_false_negative_strict_from_district_substring",
    ),
    "419": (
        "bikes",
        "easier",
        "POS",
        "changed_mobility_suggestion_from_neu_to_positive",
    ),
}


UNCERTAIN_NO_TRIPLET_IDS = {
    "30": "positive expression but aspect is implicit from context",
    "72": "positive expression but aspect is implicit from context",
    "87": "comfort is explicit, but aspect is implicit",
    "130": "check-in/out likely positive, wording is malformed",
    "246": "positive expression but aspect is implicit",
    "302": "amenity mention is weak and not clearly evaluated",
    "375": "host/person praise, aspect wording would violate no-person-name rule",
    "382": "sleep/check-in positive, aspect boundary is ambiguous",
    "397": "positive Berlin/trip context, not listing-specific enough",
    "408": "positive expression but aspect is implicit from context",
    "530": "positive overall judgment but aspect is implicit",
    "550": "fridge has mixed evaluation; sentiment boundary ambiguous",
    "612": "smalltalk, keep no triplet",
    "675": "administrative apartment mention, not clearly listing quality",
    "692": "person praise, not listing-specific enough",
    "719": "host communication likely positive, but aspect is partly implicit",
    "789": "long sentence with keys/apartment promise; needs human review",
    "808": "likely typo for super; aspect is implicit",
    "846": "host/person praise, no exact non-person aspect",
    "878": "host/person praise, no exact non-person aspect",
    "879": "host/person praise, no exact non-person aspect",
    "912": "positive organization, aspect boundary ambiguous",
    "944": "host/check-in kindness, aspect boundary ambiguous",
    "977": "counterfactual suitability statement, leave for human review",
}


def set_triplets(row: dict[str, str], aspects: str, opinions: str, sentiments: str) -> None:
    row["gold_aspect_text"] = aspects
    row["gold_opinion_text"] = opinions
    row["gold_sentiment"] = sentiments
    for column in SPAN_COLUMNS:
        row[column] = ""


def review(args: argparse.Namespace) -> Counter:
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    examples_for_human_check: list[dict[str, str]] = []

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        for column in AI_COLUMNS:
            if column not in fieldnames:
                fieldnames.insert(6, column)

        rows: list[dict[str, str]] = []
        for row in reader:
            counts["rows_seen"] += 1
            annotation_id = row.get("annotation_id", "")
            row["gold_review_status"] = "ai_reviewed_human_check_required"
            row["ai_review_status"] = "checked_by_ai"
            row["ai_review_action"] = "kept_existing_candidate"
            row["ai_review_notes"] = "KI-Review: no conservative change applied; human final check still required"

            if annotation_id in HIGH_CONFIDENCE_CORRECTIONS:
                aspects, opinions, sentiments, reason = HIGH_CONFIDENCE_CORRECTIONS[annotation_id]
                set_triplets(row, aspects, opinions, sentiments)
                row["primary_label_group"] = "NEG" if "NEG" in sentiments.split(" ; ") else "POS"
                row["label_source"] = "ai_added_review_required"
                row["ai_review_action"] = "added_triplet"
                row["ai_review_notes"] = f"KI-Review high-confidence correction: {reason}"
                counts["high_confidence_triplets_added"] += 1
            elif annotation_id in HIGH_CONFIDENCE_REVISIONS:
                aspects, opinions, sentiments, reason = HIGH_CONFIDENCE_REVISIONS[annotation_id]
                set_triplets(row, aspects, opinions, sentiments)
                row["primary_label_group"] = "NEG" if "NEG" in sentiments.split(" ; ") else "POS"
                row["label_source"] = "ai_corrected_review_required"
                row["ai_review_action"] = "corrected_triplet"
                row["ai_review_notes"] = f"KI-Review high-confidence correction: {reason}"
                counts["high_confidence_triplets_corrected"] += 1
            elif annotation_id in UNCERTAIN_NO_TRIPLET_IDS:
                row["ai_review_status"] = "needs_human_decision"
                row["ai_review_action"] = "kept_empty_but_flagged"
                row["ai_review_notes"] = f"KI-Review uncertain: {UNCERTAIN_NO_TRIPLET_IDS[annotation_id]}"
                row["difficult"] = "yes"
                examples_for_human_check.append(
                    {
                        "annotation_id": annotation_id,
                        "split": row.get("split", ""),
                        "sentence": row.get("sentence", ""),
                        "reason": UNCERTAIN_NO_TRIPLET_IDS[annotation_id],
                    }
                )
                counts["uncertain_rows_flagged"] += 1

            if row.get("gold_aspect_text", "").strip():
                counts["rows_with_triplets_after_ai_review"] += 1
            else:
                counts["rows_without_triplets_after_ai_review"] += 1
            counts[f"ai_action_{row['ai_review_action']}"] += 1
            rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = dict(sorted(counts.items()))
    summary["input"] = str(input_path)
    summary["output"] = str(output_path)
    summary["status"] = "ai_reviewed_human_check_required"
    summary["human_check_examples"] = examples_for_human_check

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an AI-reviewed copy of the 450 split.")
    parser.add_argument(
        "--input",
        default="data/annotations/span_aste_gold_review_450_split.csv",
        help="Original 450 split CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/annotations/span_aste_gold_review_450_split_ki_geprueft.csv",
        help="AI-reviewed output CSV.",
    )
    parser.add_argument(
        "--summary",
        default="data/annotations/span_aste_gold_review_450_split_ki_geprueft_summary.json",
        help="Summary JSON to write.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = review(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
