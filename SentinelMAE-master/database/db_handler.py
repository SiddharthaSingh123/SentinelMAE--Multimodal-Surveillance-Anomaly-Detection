"""
database/db_handler.py

SQLite database handler for storing and querying SentinelMAE incidents.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH    = "database/sentinel.db"
SCHEMA_PATH = "database/schema.sql"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH, schema_path: str = SCHEMA_PATH) -> None:
    """Create tables if they don't exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with open(schema_path, "r") as f:
        schema = f.read()
    conn = get_connection(db_path)
    conn.executescript(schema)
    conn.commit()
    conn.close()
    print(f"Database initialized: {db_path}")


def save_incident(
    video_path      : str,
    predicted_class : str,
    confidence      : float,
    risk_level      : str,
    risk_score      : float,
    alert           : bool,
    clip_predictions=None,
    notes           : str = "",
    db_path         : str = DB_PATH,
) -> int:
    """
    Save a detection incident to the database.

    Returns:
        incident_id (int)
    """
    conn = get_connection(db_path)
    cur  = conn.cursor()

    cur.execute("""
        INSERT INTO incidents
            (video_path, predicted_class, confidence, risk_level,
             risk_score, alert, detected_at, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        video_path,
        predicted_class,
        confidence,
        risk_level,
        risk_score,
        int(alert),
        datetime.utcnow().isoformat(),
        notes,
    ))

    incident_id = cur.lastrowid

    if clip_predictions:
        for clip in clip_predictions:
            cur.execute("""
                INSERT INTO clip_predictions
                    (incident_id, clip_idx, top1_class, top1_conf)
                VALUES (?, ?, ?, ?)
            """, (
                incident_id,
                clip["clip_idx"],
                clip["top1_class"],
                clip["top1_conf"],
            ))

    conn.commit()
    conn.close()
    return incident_id


def get_incidents(
    risk_level=None,
    alert_only : bool = False,
    limit      : int = 50,
    db_path    : str = DB_PATH,
) -> list[dict]:
    """
    Query incidents from the database.

    Args:
        risk_level : filter by level e.g. 'CRITICAL', 'HIGH'
        alert_only : if True, return only alerted incidents
        limit      : max number of results
    """
    conn = get_connection(db_path)
    cur  = conn.cursor()

    query  = "SELECT * FROM incidents WHERE 1=1"
    params = []

    if risk_level:
        query += " AND risk_level = ?"
        params.append(risk_level)

    if alert_only:
        query += " AND alert = 1"

    query += " ORDER BY detected_at DESC LIMIT ?"
    params.append(limit)

    cur.execute(query, params)
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_stats(db_path: str = DB_PATH) -> dict:
    """Return summary statistics from the database."""
    conn = get_connection(db_path)
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM incidents")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM incidents WHERE alert = 1")
    alerts = cur.fetchone()[0]

    cur.execute("""
        SELECT risk_level, COUNT(*) as count
        FROM incidents
        GROUP BY risk_level
        ORDER BY count DESC
    """)
    by_level = {row[0]: row[1] for row in cur.fetchall()}

    cur.execute("""
        SELECT predicted_class, COUNT(*) as count
        FROM incidents
        GROUP BY predicted_class
        ORDER BY count DESC
        LIMIT 5
    """)
    top_classes = {row[0]: row[1] for row in cur.fetchall()}

    conn.close()
    return {
        "total_incidents" : total,
        "total_alerts"    : alerts,
        "by_risk_level"   : by_level,
        "top_classes"     : top_classes,
    }


def print_stats(db_path: str = DB_PATH) -> None:
    stats = get_stats(db_path)
    print("\n── Database Stats ─────────────────────────────────────")
    print(f"  Total incidents : {stats['total_incidents']}")
    print(f"  Total alerts    : {stats['total_alerts']}")
    print(f"  By risk level   : {stats['by_risk_level']}")
    print(f"  Top classes     : {stats['top_classes']}")
    print("───────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    init_db()

    # Test: save a dummy incident
    inc_id = save_incident(
        video_path       = "data/raw/Fighting/video001.mp4",
        predicted_class  = "Fighting",
        confidence       = 0.76,
        risk_level       = "HIGH",
        risk_score       = 0.57,
        alert            = True,
        clip_predictions = [
            {"clip_idx": 0, "top1_class": "Fighting", "top1_conf": 0.76},
            {"clip_idx": 1, "top1_class": "Fighting", "top1_conf": 0.81},
        ],
    )
    print(f"Saved incident ID: {inc_id}")
    print_stats()