-- SentinelMAE Database Schema
-- Stores incident detections, risk scores, and video metadata

CREATE TABLE IF NOT EXISTS incidents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_path      TEXT    NOT NULL,
    predicted_class TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    risk_level      TEXT    NOT NULL,
    risk_score      REAL    NOT NULL,
    alert           INTEGER NOT NULL DEFAULT 0,  -- 0 = false, 1 = true
    detected_at     TEXT    NOT NULL,             -- ISO timestamp
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS clip_predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id INTEGER NOT NULL,
    clip_idx    INTEGER NOT NULL,
    top1_class  TEXT    NOT NULL,
    top1_conf   REAL    NOT NULL,
    FOREIGN KEY (incident_id) REFERENCES incidents(id)
);

CREATE INDEX IF NOT EXISTS idx_risk_level   ON incidents(risk_level);
CREATE INDEX IF NOT EXISTS idx_detected_at  ON incidents(detected_at);
CREATE INDEX IF NOT EXISTS idx_alert        ON incidents(alert);