"""
chatbot/query_parser.py

Parses natural language queries into structured database filters.

Examples:
    "Show me all fighting incidents"
    → {"predicted_class": "Fighting"}

    "Show critical alerts from today"
    → {"risk_level": "CRITICAL", "alert_only": True}

    "How many robberies were detected?"
    → {"predicted_class": "Robbery", "count_only": True}
"""

import re
from datetime import datetime


# ── Keyword maps ───────────────────────────────────────────────────────────

CLASS_KEYWORDS = {
    "Abuse"         : ["abuse", "abusing"],
    "Arrest"        : ["arrest", "arrested"],
    "Arson"         : ["arson", "fire", "burning"],
    "Assault"       : ["assault", "assaulting"],
    "Burglary"      : ["burglary", "burgling", "break in", "breaking in"],
    "Explosion"     : ["explosion", "explode", "blast"],
    "Fighting"      : ["fight", "fighting", "brawl"],
    "NormalVideos"  : ["normal", "safe", "no threat"],
    "RoadAccidents" : ["road", "accident", "crash", "vehicle"],
    "Robbery"       : ["robbery", "robbing", "rob"],
    "Shooting"      : ["shoot", "shooting", "gun", "gunshot"],
    "Shoplifting"   : ["shoplift", "shoplifting", "stealing from store"],
    "Stealing"      : ["steal", "stealing", "theft"],
    "Vandalism"     : ["vandal", "vandalism", "graffiti", "damage"],
}

RISK_KEYWORDS = {
    "CRITICAL" : ["critical", "immediate", "emergency"],
    "HIGH"     : ["high", "violent", "dangerous"],
    "MEDIUM"   : ["medium", "moderate", "property"],
    "LOW"      : ["low", "minor", "notable"],
    "NORMAL"   : ["normal", "safe", "no risk"],
}


def parse_query(query: str) -> dict:
    """
    Parse a natural language query into a filter dict.

    Returns a dict with any of these keys:
        predicted_class : str
        risk_level      : str
        alert_only      : bool
        count_only      : bool
        limit           : int
    """
    q = query.lower().strip()
    filters = {}

    # Detect class
    for class_name, keywords in CLASS_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            filters["predicted_class"] = class_name
            break

    # Detect risk level
    for level, keywords in RISK_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            filters["risk_level"] = level
            break

    # Detect alert filter
    if any(w in q for w in ["alert", "urgent", "warned"]):
        filters["alert_only"] = True

    # Detect count query
    if any(w in q for w in ["how many", "count", "total", "number of"]):
        filters["count_only"] = True

    # Detect limit
    match = re.search(r"(last|top|show)\s+(\d+)", q)
    if match:
        filters["limit"] = int(match.group(2))

    return filters


def explain_filters(filters: dict) -> str:
    """Return a human-readable explanation of parsed filters."""
    if not filters:
        return "Showing all incidents."

    parts = []
    if "predicted_class" in filters:
        parts.append(f"class = {filters['predicted_class']}")
    if "risk_level" in filters:
        parts.append(f"risk = {filters['risk_level']}")
    if filters.get("alert_only"):
        parts.append("alerts only")
    if filters.get("count_only"):
        parts.append("count only")
    if "limit" in filters:
        parts.append(f"limit = {filters['limit']}")

    return "Filters: " + ", ".join(parts)


if __name__ == "__main__":
    test_queries = [
        "Show me all fighting incidents",
        "How many robberies were detected?",
        "Show critical alerts",
        "Last 10 shooting incidents",
        "Any high risk vandalism?",
        "Show all normal videos",
    ]

    for q in test_queries:
        filters = parse_query(q)
        print(f"Query   : {q}")
        print(f"Parsed  : {filters}")
        print(f"Explain : {explain_filters(filters)}")
        print()