#!/usr/bin/env python3
"""Aggregate Airbnb Span-ASTE triplets to listing-level analysis features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


LISTING_JOIN_FIELDS = [
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "property_type",
    "room_type",
    "accommodates",
    "price",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
]


def load_listing_features(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        return {row.get("id", ""): row for row in reader if row.get("id")}


def top_items(counter: Counter, limit: int) -> str:
    return " ; ".join(f"{key}:{count}" for key, count in counter.most_common(limit))


def sentiment_score(pos: int, neg: int, total: int) -> str:
    if total == 0:
        return ""
    return f"{(pos - neg) / total:.4f}"


def aggregate(args: argparse.Namespace) -> Counter:
    input_path = Path(args.input)
    listings_path = Path(args.listings)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    listings = load_listing_features(listings_path)
    by_listing: dict[str, dict[str, object]] = {}
    counts: Counter = Counter()

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            counts["triplets_seen"] += 1
            if args.usable_only and row.get("usable_for_analysis") != "yes":
                counts["triplets_skipped_not_usable"] += 1
                continue

            listing_id = row.get("listing_id", "")
            if not listing_id:
                counts["triplets_skipped_missing_listing_id"] += 1
                continue

            listing = by_listing.setdefault(
                listing_id,
                {
                    "sentiments": Counter(),
                    "topics": Counter(),
                    "aspects": Counter(),
                    "positive_topics": Counter(),
                    "negative_topics": Counter(),
                    "positive_aspects": Counter(),
                    "negative_aspects": Counter(),
                    "review_ids": set(),
                    "sentence_ids": set(),
                },
            )

            sentiment = row.get("sentiment", "")
            topic = row.get("topic_category", "") or row.get("aspect_category", "") or "other/unknown"
            aspect = row.get("normalized_aspect", "") or row.get("aspect", "") or "unknown"
            review_id = row.get("review_id", "")
            sentence_key = f"{review_id}:{row.get('sentence_id', '')}"

            listing["sentiments"][sentiment] += 1
            listing["topics"][topic] += 1
            listing["aspects"][aspect] += 1
            listing["review_ids"].add(review_id)
            listing["sentence_ids"].add(sentence_key)
            if sentiment == "POS":
                listing["positive_topics"][topic] += 1
                listing["positive_aspects"][aspect] += 1
            elif sentiment == "NEG":
                listing["negative_topics"][topic] += 1
                listing["negative_aspects"][aspect] += 1

            counts["triplets_used"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    output_fields = [
        "listing_id",
        "review_count_with_triplets",
        "sentence_count_with_triplets",
        "triplet_count",
        "sentiment_pos",
        "sentiment_neu",
        "sentiment_neg",
        "sentiment_pos_share",
        "sentiment_neu_share",
        "sentiment_neg_share",
        "sentiment_net_score",
        "top_topics",
        "top_positive_topics",
        "top_negative_topics",
        "top_aspects",
        "top_positive_aspects",
        "top_negative_aspects",
        *LISTING_JOIN_FIELDS,
    ]

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fields)
        writer.writeheader()
        for listing_id in sorted(by_listing, key=lambda value: int(value) if value.isdigit() else value):
            listing = by_listing[listing_id]
            sentiments: Counter = listing["sentiments"]
            total = sum(sentiments.values())
            pos = sentiments.get("POS", 0)
            neu = sentiments.get("NEU", 0)
            neg = sentiments.get("NEG", 0)
            listing_features = listings.get(listing_id, {})
            row = {
                "listing_id": listing_id,
                "review_count_with_triplets": len(listing["review_ids"]),
                "sentence_count_with_triplets": len(listing["sentence_ids"]),
                "triplet_count": total,
                "sentiment_pos": pos,
                "sentiment_neu": neu,
                "sentiment_neg": neg,
                "sentiment_pos_share": f"{pos / total:.4f}" if total else "",
                "sentiment_neu_share": f"{neu / total:.4f}" if total else "",
                "sentiment_neg_share": f"{neg / total:.4f}" if total else "",
                "sentiment_net_score": sentiment_score(pos, neg, total),
                "top_topics": top_items(listing["topics"], args.top_n),
                "top_positive_topics": top_items(listing["positive_topics"], args.top_n),
                "top_negative_topics": top_items(listing["negative_topics"], args.top_n),
                "top_aspects": top_items(listing["aspects"], args.top_n),
                "top_positive_aspects": top_items(listing["positive_aspects"], args.top_n),
                "top_negative_aspects": top_items(listing["negative_aspects"], args.top_n),
            }
            for field in LISTING_JOIN_FIELDS:
                row[field] = listing_features.get(field, "")
            writer.writerow(row)

    counts["listings_written"] = len(by_listing)
    counts["listing_features_joined"] = sum(1 for listing_id in by_listing if listing_id in listings)
    summary = dict(sorted(counts.items()))
    summary["input"] = str(input_path)
    summary["output"] = str(output_path)
    summary["listings"] = str(listings_path)
    summary["usable_only"] = args.usable_only

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate flat Span-ASTE triplets to listing-level features."
    )
    parser.add_argument(
        "--input",
        default="data/processed/airbnb_aste_triplets_1000_ergaenzt_flat.csv",
        help="Triplet-level CSV.",
    )
    parser.add_argument(
        "--listings",
        default="listings_clean.csv",
        help="Listing feature table with an id column.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/airbnb_aste_listing_aggregation_1000_sample.csv",
        help="Listing-level output CSV.",
    )
    parser.add_argument(
        "--summary",
        default="data/processed/airbnb_aste_listing_aggregation_1000_sample_summary.json",
        help="Summary JSON to write.",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--include-review-qc",
        dest="usable_only",
        action="store_false",
        help="Include triplets marked qc_status=review in addition to usable keep rows.",
    )
    parser.set_defaults(usable_only=True)
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = aggregate(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
