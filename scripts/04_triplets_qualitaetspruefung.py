#!/usr/bin/env python3
"""Add rule-based quality-control labels to parsed Span-ASTE triplets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PUNCT_EDGE_RE = re.compile(r"^[^\w]+|[^\w]+$")
SPACE_RE = re.compile(r"\s+")

PERSON_NAMES = {
    "brita",
    "britta",
    "drawe",
    "friederich",
    "marcus",
    "matthias",
    "max",
}

DROP_ASPECTS = {
    "airbnb": "platform_name",
    "berlin": "city_name",
    "expat": "not_listing_aspect",
    "flight": "travel_context",
    "friend": "person_or_context_not_aspect",
    "landlord horror stories": "story_context_not_aspect",
    "memory": "too_contextual",
    "partiicpate": "travel_context",
    "people": "too_generic",
    "staying": "action_not_aspect",
    "stocked": "opinion_extracted_as_aspect",
    "time": "too_generic",
    "waiting": "action_not_aspect",
    "wich": "extraction_noise",
}

NORMALIZE_MAP = {
    "accommodation": ("apartment", "accommodation"),
    "appartement": ("apartment", "accommodation"),
    "appartment": ("apartment", "accommodation"),
    "apartment": ("apartment", "accommodation"),
    "apartment block": ("building", "building"),
    "apartment self": ("apartment", "accommodation"),
    "apt": ("apartment", "accommodation"),
    "family rental": ("apartment", "accommodation"),
    "flat": ("apartment", "accommodation"),
    "home": ("apartment", "accommodation"),
    "house": ("apartment", "accommodation"),
    "listing": ("apartment", "accommodation"),
    "loft": ("apartment", "accommodation"),
    "place": ("apartment", "accommodation"),
    "property": ("apartment", "accommodation"),
    "room": ("room", "accommodation"),
    "studio": ("studio", "accommodation"),
    "sublet": ("apartment", "accommodation"),
    "temp home": ("apartment", "accommodation"),
    "area": ("location", "location"),
    "areas": ("location", "location"),
    "location": ("location", "location"),
    "neighborhood": ("neighborhood", "location"),
    "neighborhoods": ("neighborhood", "location"),
    "neighbourhood": ("neighborhood", "location"),
    "neighbourhoods": ("neighborhood", "location"),
    "residential area": ("neighborhood", "location"),
    "zone": ("location", "location"),
    "s-bahn": ("public_transport", "transport"),
    "subways": ("public_transport", "transport"),
    "tram": ("public_transport", "transport"),
    "tram line": ("public_transport", "transport"),
    "tram ride": ("public_transport", "transport"),
    "transport links": ("public_transport", "transport"),
    "u-bahn": ("public_transport", "transport"),
    "ground transportation": ("public_transport", "transport"),
    "access": ("access", "transport"),
    "connections": ("connections", "transport"),
    "metro": ("public_transport", "transport"),
    "bathroom": ("bathroom", "amenities"),
    "bed": ("bed", "comfort"),
    "bed upstairs": ("bed", "comfort"),
    "bedroom": ("bedroom", "comfort"),
    "beds": ("bed", "comfort"),
    "blankets": ("bedding", "amenities"),
    "couch": ("couch", "comfort"),
    "heating": ("heating", "amenities"),
    "hot water": ("hot_water", "amenities"),
    "king bed": ("bed", "comfort"),
    "kitchen": ("kitchen", "amenities"),
    "kitchen area": ("kitchen", "amenities"),
    "linen": ("bedding", "amenities"),
    "sheets": ("bedding", "amenities"),
    "sofa in living room": ("sofa", "comfort"),
    "towels": ("towels", "amenities"),
    "wifi": ("wifi", "amenities"),
    "baby cot bed": ("baby_cot", "amenities"),
    "dishwasher": ("dishwasher", "amenities"),
    "freezer": ("freezer", "amenities"),
    "touch lamps": ("lighting", "amenities"),
    "availabe hairdryer": ("hairdryer", "amenities"),
    "chair": ("chair", "amenities"),
    "high chair": ("chair", "amenities"),
    "pilot light": ("heating", "amenities"),
    "radio": ("radio", "amenities"),
    "tea": ("tea_coffee", "amenities"),
    "furniture": ("furniture", "interior"),
    "furnitures": ("furniture", "interior"),
    "light": ("light", "interior"),
    "livingroom": ("living_room", "interior"),
    "loungespot": ("lounge_spot", "interior"),
    "space": ("space", "interior"),
    "windows": ("windows", "interior"),
    "view": ("view", "view"),
    "views": ("view", "view"),
    "balcony": ("balcony", "outdoor"),
    "canal": ("surroundings", "location"),
    "channel": ("surroundings", "location"),
    "park": ("park", "location"),
    "parks": ("park", "location"),
    "patio": ("patio", "outdoor"),
    "street": ("street", "location"),
    "steet": ("street", "location"),
    "building": ("building", "building"),
    "construction": ("construction", "building"),
    "building work": ("construction", "building"),
    "door": ("door", "building"),
    "noise": ("noise", "noise"),
    "price": ("price", "value"),
    "quality": ("quality", "value"),
    "value": ("value", "value"),
    "check in": ("check_in", "check_in"),
    "check-in": ("check_in", "check_in"),
    "check-in time": ("check_in", "check_in"),
    "check-out": ("check_out", "check_in"),
    "checkin": ("check_in", "check_in"),
    "communication": ("communication", "host_communication"),
    "contact": ("communication", "host_communication"),
    "directions": ("directions", "host_communication"),
    "host": ("host", "host_communication"),
    "hosts": ("host", "host_communication"),
    "information": ("information", "host_communication"),
    "informations": ("information", "host_communication"),
    "info": ("information", "host_communication"),
    "instructions": ("instructions", "host_communication"),
    "key pick up": ("key_pickup", "check_in"),
    "keyholder": ("keyholder", "host_communication"),
    "landlady": ("host", "host_communication"),
    "management company": ("management", "host_communication"),
    "management team": ("management", "host_communication"),
    "owner": ("host", "host_communication"),
    "owners": ("host", "host_communication"),
    "personal recommendations": ("recommendations", "host_communication"),
    "tips": ("recommendations", "host_communication"),
    "welcome packet": ("welcome_packet", "host_communication"),
    "advice": ("recommendations", "host_communication"),
    "feedback": ("feedback", "host_communication"),
    "organisation": ("organization", "host_communication"),
    "reservation": ("reservation", "booking"),
    "appointments": ("appointments", "booking"),
    "cafes": ("cafes_restaurants", "surroundings_food"),
    "cafe": ("cafes_restaurants", "surroundings_food"),
    "cafés": ("cafes_restaurants", "surroundings_food"),
    "bars": ("bars", "surroundings_food"),
    "restaurants": ("restaurants", "surroundings_food"),
    "food": ("food", "surroundings_food"),
    "food market": ("market", "surroundings_food"),
    "food shops": ("shops", "surroundings_food"),
    "grocery stores": ("shops", "surroundings_food"),
    "market": ("market", "surroundings_food"),
    "markets": ("market", "surroundings_food"),
    "organic supermarket": ("supermarket", "surroundings_food"),
    "supermarkets": ("supermarket", "surroundings_food"),
    "shops": ("shops", "surroundings_food"),
    "shopping area": ("shops", "surroundings_food"),
    "turkish market": ("market", "surroundings_food"),
    "breakfast place": ("cafes_restaurants", "surroundings_food"),
    "brunch": ("cafes_restaurants", "surroundings_food"),
    "burger place": ("restaurants", "surroundings_food"),
    "coffee": ("coffee", "surroundings_food"),
    "coffee places": ("coffee", "surroundings_food"),
    "convenience stores": ("shops", "surroundings_food"),
    "eating places": ("restaurants", "surroundings_food"),
    "organic stuff": ("organic_food", "surroundings_food"),
    "sushi": ("restaurants", "surroundings_food"),
    "vietnamese food": ("restaurants", "surroundings_food"),
    "wine": ("drinks", "surroundings_food"),
    "bikes": ("bikes", "mobility"),
    "playground": ("playground", "family"),
    "playgrounds": ("playground", "family"),
    "kids": ("kids", "family"),
    "baby": ("baby", "family"),
    "stay": ("stay", "overall_experience"),
    "spot": ("apartment", "accommodation"),
    "mini trip": ("stay", "overall_experience"),
    "walk": ("walkability", "location"),
    "walks": ("walkability", "location"),
    "riverside walks": ("walkability", "location"),
    "atms": ("atms", "location"),
    "lines": ("public_transport", "transport"),
    "pine lanes": ("street", "location"),
    "tree-lines": ("street", "location"),
    "neighbors": ("neighbors", "location"),
    "provisions": ("provisions", "amenities"),
    "hub": ("stay", "overall_experience"),
    "nest": ("apartment", "accommodation"),
    "loft bed": ("bed", "comfort"),
    "mezzanine bed": ("bed", "comfort"),
    "matthias studio": ("studio", "accommodation"),
    "variety of food": ("food", "surroundings_food"),
    "bio supermarkets": ("supermarket", "surroundings_food"),
    "drinks": ("drinks", "surroundings_food"),
    "baker": ("bakery", "surroundings_food"),
    "cakes": ("cakes", "surroundings_food"),
    "meal": ("meal", "surroundings_food"),
    "meals": ("meal", "surroundings_food"),
    "croissants": ("bakery", "surroundings_food"),
    "menu": ("restaurants", "surroundings_food"),
    "turquish market": ("market", "surroundings_food"),
    "doorstep markets": ("market", "surroundings_food"),
    "street markets": ("market", "surroundings_food"),
    "clubs": ("clubs", "surroundings"),
    "maps": ("maps", "host_communication"),
    "bouncer": ("bouncer", "family"),
}


def clean_aspect(value: str) -> str:
    value = SPACE_RE.sub(" ", value or "").strip()
    value = value.replace("’", "'").replace("`", "'")
    value = PUNCT_EDGE_RE.sub("", value)
    value = SPACE_RE.sub(" ", value).strip().lower()
    return value


def classify_aspect(raw_aspect: str) -> tuple[str, str, str, str]:
    clean = clean_aspect(raw_aspect)
    if not clean:
        return "", "", "drop", "empty_aspect"

    if clean in PERSON_NAMES:
        return clean, "person_name", "drop", "person_or_host_name"

    if clean in DROP_ASPECTS:
        return clean, "exclude", "drop", DROP_ASPECTS[clean]

    if clean in NORMALIZE_MAP:
        normalized, category = NORMALIZE_MAP[clean]
        return normalized, category, "keep", "mapped_domain_aspect"

    if any(name in clean.split() for name in PERSON_NAMES):
        return clean, "possible_person_name", "review", "contains_person_name"

    if raw_aspect[:1].isupper() and clean not in NORMALIZE_MAP:
        return clean, "unknown", "review", "capitalized_possible_name_or_place"

    if len(clean) <= 2:
        return clean, "unknown", "review", "very_short_unknown_aspect"

    return clean, "unknown", "review", "unmapped_aspect"


def qc_triplets(args: argparse.Namespace) -> Counter:
    input_path = Path(args.input)
    output_path = Path(args.output)
    clean_output_path = Path(args.clean_output)
    aspect_rules_path = Path(args.aspect_rules)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_output_path.parent.mkdir(parents=True, exist_ok=True)
    aspect_rules_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    aspect_counts: Counter = Counter()
    aspect_examples: dict[str, str] = {}
    rows = []

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []
        for row in reader:
            normalized, category, qc_status, qc_reason = classify_aspect(row["aspect"])
            row.update(
                {
                    "normalized_aspect": normalized,
                    "aspect_category": category,
                    "qc_status": qc_status,
                    "qc_reason": qc_reason,
                    "usable_for_analysis": "yes" if qc_status == "keep" else "no",
                }
            )
            rows.append(row)
            counts[f"qc_status_{qc_status}"] += 1
            counts[f"aspect_category_{category}"] += 1
            aspect_counts[row["aspect"]] += 1
            aspect_examples.setdefault(row["aspect"], row["sentence"])

    output_fields = [
        *fieldnames,
        "normalized_aspect",
        "aspect_category",
        "qc_status",
        "qc_reason",
        "usable_for_analysis",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    with clean_output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows([row for row in rows if row["qc_status"] == "keep"])

    rule_rows = []
    for raw_aspect, count in aspect_counts.most_common():
        normalized, category, qc_status, qc_reason = classify_aspect(raw_aspect)
        rule_rows.append(
            {
                "aspect": raw_aspect,
                "count": count,
                "normalized_aspect": normalized,
                "aspect_category": category,
                "qc_status": qc_status,
                "qc_reason": qc_reason,
                "example_sentence": aspect_examples[raw_aspect],
            }
        )

    with aspect_rules_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "aspect",
                "count",
                "normalized_aspect",
                "aspect_category",
                "qc_status",
                "qc_reason",
                "example_sentence",
            ],
        )
        writer.writeheader()
        writer.writerows(rule_rows)

    counts["triplets_total"] = len(rows)
    counts["unique_aspects"] = len(aspect_counts)
    counts["unique_normalized_aspects"] = len(
        {row["normalized_aspect"] for row in rows if row["normalized_aspect"]}
    )
    counts["usable_triplets"] = counts["qc_status_keep"]

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(dict(sorted(counts.items())), summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark Span-ASTE triplets as keep/review/drop and normalize aspects."
    )
    parser.add_argument(
        "--input",
        default="Span-ASTE/data_airbnb/03_triplets_modell_roh_1000_saetze.csv",
        help="Parsed Span-ASTE triplet CSV.",
    )
    parser.add_argument(
        "--output",
        default="Span-ASTE/data_airbnb/04_triplets_qualitaetspruefung_1000_saetze.csv",
        help="Triplet-level QC CSV to write.",
    )
    parser.add_argument(
        "--clean-output",
        default="Span-ASTE/data_airbnb/05_triplets_bereinigt_1000_saetze.csv",
        help="Triplet-level CSV with only qc_status=keep rows.",
    )
    parser.add_argument(
        "--aspect-rules",
        default="Span-ASTE/data_airbnb/04_aspekt_regeln_qualitaetspruefung.csv",
        help="Unique-aspect QC table to write.",
    )
    parser.add_argument(
        "--summary",
        default="Span-ASTE/data_airbnb/04_triplets_qualitaetspruefung_1000_saetze_zusammenfassung.json",
        help="Summary JSON for the QC run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result_counts = qc_triplets(parse_args())
    print(json.dumps(dict(sorted(result_counts.items())), ensure_ascii=False, indent=2))
