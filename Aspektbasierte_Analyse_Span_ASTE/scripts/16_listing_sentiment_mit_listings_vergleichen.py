#!/usr/bin/env python3
"""Compare listing-level ASTE sentiment features with Airbnb listing metadata."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "triplet_count",
    "sentiment_pos_share",
    "sentiment_neu_share",
    "sentiment_neg_share",
    "sentiment_net_score",
    "price",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
]

COMPARISON_COLUMNS = [
    "price",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
]

GROUP_COLUMNS = ["neighbourhood_group_cleansed", "room_type"]


def safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def pearson(left: list[float], right: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(left, right) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if not denom_x or not denom_y:
        return None
    return numerator / (denom_x * denom_y)


def parse_counter_text(value: str) -> Counter:
    counter: Counter = Counter()
    for item in str(value or "").split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        key, raw_count = item.rsplit(":", 1)
        try:
            counter[key.strip()] += int(raw_count.strip())
        except ValueError:
            continue
    return counter


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_chart_value(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}".replace(",", ".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def is_missing_group_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none"}


def svg_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    x_label: str,
    width: int = 920,
    height: int = 560,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin_left = 220
    margin_right = 40
    margin_top = 54
    margin_bottom = 42
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 1
    row_height = plot_height / max(len(labels), 1)
    palette = ["#386fa4", "#59a14f", "#f28e2b", "#b07aa1", "#e15759", "#4e79a7"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="30" font-family="Arial" font-size="20" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{margin_left}" y="{height - 10}" font-family="Arial" font-size="12" fill="#555">{html.escape(x_label)}</text>',
    ]

    for index, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + index * row_height + row_height * 0.18
        bar_height = max(row_height * 0.58, 12)
        bar_width = 0 if max_value == 0 else (value / max_value) * plot_width
        color = palette[index % len(palette)]
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + bar_height * 0.68:.1f}" '
            f'text-anchor="end" font-family="Arial" font-size="12" fill="#222">'
            f'{html.escape(label[:34])}</text>'
        )
        parts.append(
            f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{margin_left + bar_width + 6:.1f}" y="{y + bar_height * 0.68:.1f}" '
            f'font-family="Arial" font-size="12" fill="#222">{format_chart_value(value)}</text>'
        )

    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_scatter(
    path: Path,
    title: str,
    points: list[tuple[float, float]],
    x_label: str,
    y_label: str,
    width: int = 760,
    height: int = 560,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin_left = 74
    margin_right = 34
    margin_top = 56
    margin_bottom = 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        min_y -= 1
        max_y += 1

    def sx(value: float) -> float:
        return margin_left + (value - min_x) / (max_x - min_x) * plot_width

    def sy(value: float) -> float:
        return margin_top + plot_height - (value - min_y) / (max_y - min_y) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="30" font-family="Arial" font-size="20" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333"/>',
        f'<text x="{margin_left + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(x_label)}</text>',
        f'<text x="18" y="{margin_top + plot_height / 2}" transform="rotate(-90 18 {margin_top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(y_label)}</text>',
        f'<text x="{margin_left}" y="{margin_top + plot_height + 18}" font-family="Arial" font-size="11" fill="#555">{min_x:.2f}</text>',
        f'<text x="{margin_left + plot_width}" y="{margin_top + plot_height + 18}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{max_x:.2f}</text>',
        f'<text x="{margin_left - 8}" y="{margin_top + plot_height}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{min_y:.2f}</text>',
        f'<text x="{margin_left - 8}" y="{margin_top + 4}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{max_y:.2f}</text>',
    ]
    for x_value, y_value in points:
        parts.append(
            f'<circle cx="{sx(x_value):.1f}" cy="{sy(y_value):.1f}" r="4.2" '
            'fill="#386fa4" fill-opacity="0.62"/>'
        )
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, object]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if args.min_triplets:
        df_analysis = df[df["triplet_count"].fillna(0) >= args.min_triplets].copy()
    else:
        df_analysis = df.copy()

    correlation_rows: list[dict[str, object]] = []
    for column in COMPARISON_COLUMNS:
        if column not in df_analysis.columns:
            continue
        valid = df_analysis[["sentiment_net_score", column]].dropna()
        correlation = pearson(
            valid["sentiment_net_score"].tolist(),
            valid[column].tolist(),
        )
        correlation_rows.append(
            {
                "comparison": column,
                "n": len(valid),
                "pearson_r_with_sentiment_net_score": "" if correlation is None else round(correlation, 4),
            }
        )
    write_csv(
        output_dir / "korrelationen_sentiment_listing_merkmale.csv",
        correlation_rows,
        ["comparison", "n", "pearson_r_with_sentiment_net_score"],
    )

    group_rows: list[dict[str, object]] = []
    for group_column in GROUP_COLUMNS:
        if group_column not in df_analysis.columns:
            continue
        grouped = (
            df_analysis.groupby(group_column, dropna=False)
            .agg(
                listings=("listing_id", "count"),
                triplets=("triplet_count", "sum"),
                mean_net_sentiment=("sentiment_net_score", "mean"),
                mean_rating=("review_scores_rating", "mean"),
                mean_price=("price", "mean"),
            )
            .reset_index()
            .sort_values(["listings", "triplets"], ascending=False)
        )
        for row in grouped.to_dict(orient="records"):
            row["group_column"] = group_column
            row["group_value"] = row.pop(group_column)
            group_rows.append(row)
    write_csv(
        output_dir / "gruppenvergleich_lage_roomtype.csv",
        group_rows,
        [
            "group_column",
            "group_value",
            "listings",
            "triplets",
            "mean_net_sentiment",
            "mean_rating",
            "mean_price",
        ],
    )

    topic_counter: Counter = Counter()
    positive_topic_counter: Counter = Counter()
    negative_topic_counter: Counter = Counter()
    for _, row in df_analysis.iterrows():
        topic_counter.update(parse_counter_text(row.get("top_topics", "")))
        positive_topic_counter.update(parse_counter_text(row.get("top_positive_topics", "")))
        negative_topic_counter.update(parse_counter_text(row.get("top_negative_topics", "")))

    topic_rows = []
    all_topics = set(topic_counter) | set(positive_topic_counter) | set(negative_topic_counter)
    for topic in sorted(all_topics):
        topic_rows.append(
            {
                "topic": topic,
                "total_mentions": topic_counter[topic],
                "positive_mentions": positive_topic_counter[topic],
                "negative_mentions": negative_topic_counter[topic],
            }
        )
    topic_rows.sort(key=lambda row: row["total_mentions"], reverse=True)
    write_csv(
        output_dir / "themen_haeufigkeiten.csv",
        topic_rows,
        ["topic", "total_mentions", "positive_mentions", "negative_mentions"],
    )

    top_topics = topic_rows[: args.top_n]
    if top_topics:
        svg_bar_chart(
            output_dir / "grafik_top_themen.svg",
            "Top-Themen in ASTE-Triplets",
            [str(row["topic"]) for row in top_topics],
            [float(row["total_mentions"]) for row in top_topics],
            "Triplet-Nennungen",
        )

    group_plot_rows = [
        row
        for row in group_rows
        if row["group_column"] == "neighbourhood_group_cleansed"
        and not is_missing_group_value(row["group_value"])
        and safe_float(row["mean_net_sentiment"]) is not None
    ][: args.top_n]
    if group_plot_rows:
        svg_bar_chart(
            output_dir / "grafik_sentiment_nach_lage.svg",
            "Netto-Sentiment nach Lagegruppe",
            [str(row["group_value"]) for row in group_plot_rows],
            [float(row["mean_net_sentiment"]) for row in group_plot_rows],
            "Durchschnittlicher Net-Sentiment-Score",
        )

    scatter_points = [
        (float(row["review_scores_rating"]), float(row["sentiment_net_score"]))
        for _, row in df_analysis.dropna(
            subset=["review_scores_rating", "sentiment_net_score"]
        ).iterrows()
    ]
    if scatter_points:
        svg_scatter(
            output_dir / "grafik_rating_vs_sentiment.svg",
            "Bewertung vs. ASTE-Netto-Sentiment",
            scatter_points,
            "review_scores_rating",
            "sentiment_net_score",
        )

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "listings_seen": int(len(df)),
        "listings_after_min_triplets_filter": int(len(df_analysis)),
        "min_triplets": args.min_triplets,
        "correlations_written": len(correlation_rows),
        "group_rows_written": len(group_rows),
        "topic_rows_written": len(topic_rows),
        "graphics": [
            str(output_dir / "grafik_top_themen.svg"),
            str(output_dir / "grafik_sentiment_nach_lage.svg"),
            str(output_dir / "grafik_rating_vs_sentiment.svg"),
        ],
    }
    summary_path = output_dir / "listing_sentiment_vergleich_summary.json"
    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ASTE listing aggregation with rating, price and location."
    )
    parser.add_argument(
        "--input",
        default="Ergebnisse_50kStichprobe/listing_auswertung_50k.csv",
        help="Listing-level ASTE aggregation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="Ergebnisse_50kStichprobe/vergleich_rating_preis_lage",
        help="Directory for comparison tables and SVG charts.",
    )
    parser.add_argument(
        "--min-triplets",
        type=int,
        default=3,
        help="Minimum triplets required for listing-level comparison.",
    )
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), ensure_ascii=False, indent=2))
