"""
chatbot/retrieval.py

Connects query_parser + db_handler to answer natural language questions
about detected incidents.

Usage:
    python chatbot/retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot.query_parser import parse_query, explain_filters
from database.db_handler import init_db, get_incidents, get_stats


def answer_query(query: str, db_path: str = "database/sentinel.db") -> str:
    """
    Takes a natural language query and returns a formatted answer.

    Args:
        query   : user's question in plain English
        db_path : path to SQLite database

    Returns:
        Formatted string response
    """
    filters = parse_query(query)

    # Stats query
    if "how many" in query.lower() or filters.get("count_only"):
        incidents = get_incidents(
            risk_level = filters.get("risk_level"),
            alert_only = filters.get("alert_only", False),
            limit      = 1000,
            db_path    = db_path,
        )
        # Filter by class if specified
        if "predicted_class" in filters:
            incidents = [i for i in incidents
                         if i["predicted_class"] == filters["predicted_class"]]
        count = len(incidents)
        cls   = filters.get("predicted_class", "total")
        return f"Found {count} incident(s) for: {cls}"

    # General stats
    if any(w in query.lower() for w in ["stats", "summary", "overview", "dashboard"]):
        stats = get_stats(db_path)
        lines = [
            f"Total incidents : {stats['total_incidents']}",
            f"Total alerts    : {stats['total_alerts']}",
            f"By risk level   : {stats['by_risk_level']}",
            f"Top classes     : {stats['top_classes']}",
        ]
        return "\n".join(lines)

    # Incident listing
    incidents = get_incidents(
        risk_level = filters.get("risk_level"),
        alert_only = filters.get("alert_only", False),
        limit      = filters.get("limit", 10),
        db_path    = db_path,
    )

    # Filter by class if specified
    if "predicted_class" in filters:
        incidents = [i for i in incidents
                     if i["predicted_class"] == filters["predicted_class"]]

    if not incidents:
        return f"No incidents found. ({explain_filters(filters)})"

    lines = [f"Found {len(incidents)} incident(s) — {explain_filters(filters)}:"]
    for inc in incidents:
        alert_tag = "⚠" if inc["alert"] else " "
        lines.append(
            f"  {alert_tag} [{inc['detected_at'][:19]}]  "
            f"{inc['predicted_class']:15s}  "
            f"Risk: {inc['risk_level']:8s}  "
            f"Conf: {inc['confidence']*100:.1f}%  "
            f"→ {inc['video_path']}"
        )
    return "\n".join(lines)


def chat(db_path: str = "database/sentinel.db") -> None:
    """Interactive chat loop for querying incidents."""
    print("\n── SentinelMAE Chatbot ────────────────────────────────")
    print("  Ask questions about detected incidents.")
    print("  Type 'quit' to exit.\n")
    print("  Examples:")
    print("    - Show me all fighting incidents")
    print("    - How many robberies were detected?")
    print("    - Show critical alerts")
    print("    - Give me a summary")
    print("───────────────────────────────────────────────────────\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        response = answer_query(query, db_path)
        print(f"\nBot: {response}\n")


if __name__ == "__main__":
    init_db()
    chat()