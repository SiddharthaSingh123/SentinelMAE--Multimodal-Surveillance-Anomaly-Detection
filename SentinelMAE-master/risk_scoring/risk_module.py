"""
risk_scoring/risk_module.py

Converts model predictions into a structured risk score and alert level.

Risk Levels:
    CRITICAL  — immediate threat (shooting, explosion, arson)
    HIGH      — violent crime (assault, fighting, robbery, abuse)
    MEDIUM    — property crime (burglary, vandalism, stealing, shoplifting)
    LOW       — non-criminal but notable (arrest, road accident)
    NORMAL    — no threat detected
"""

from dataclasses import dataclass


# ── Risk configuration ─────────────────────────────────────────────────────

RISK_MAP = {
    "Shooting"      : ("CRITICAL", 1.00),
    "Explosion"     : ("CRITICAL", 0.95),
    "Arson"         : ("CRITICAL", 0.90),
    "Assault"       : ("HIGH",     0.80),
    "Fighting"      : ("HIGH",     0.75),
    "Robbery"       : ("HIGH",     0.75),
    "Abuse"         : ("HIGH",     0.70),
    "Burglary"      : ("MEDIUM",   0.55),
    "Vandalism"     : ("MEDIUM",   0.50),
    "Stealing"      : ("MEDIUM",   0.45),
    "Shoplifting"   : ("MEDIUM",   0.40),
    "Arrest"        : ("LOW",      0.25),
    "RoadAccidents" : ("LOW",      0.20),
    "NormalVideos"  : ("NORMAL",   0.00),
}

LEVEL_COLORS = {
    "CRITICAL" : "🔴",
    "HIGH"     : "🟠",
    "MEDIUM"   : "🟡",
    "LOW"      : "🔵",
    "NORMAL"   : "🟢",
}


@dataclass
class RiskResult:
    event_class   : str
    risk_level    : str
    base_score    : float
    final_score   : float
    confidence    : float
    alert         : bool
    description   : str
    color         : str


def compute_risk(predicted_class: str, confidence: float) -> RiskResult:
    """
    Compute a risk score from a model prediction.

    Args:
        predicted_class : class name string (must match RISK_MAP keys)
        confidence      : model confidence in [0, 1]

    Returns:
        RiskResult dataclass with full risk breakdown
    """
    if predicted_class not in RISK_MAP:
        predicted_class = "NormalVideos"

    risk_level, base_score = RISK_MAP[predicted_class]

    # Final score = base severity × model confidence
    final_score = round(base_score * confidence, 4)

    alert = risk_level in ("CRITICAL", "HIGH")

    descriptions = {
        "CRITICAL" : f"IMMEDIATE THREAT detected: {predicted_class}. Alert authorities.",
        "HIGH"     : f"Violent activity detected: {predicted_class}. Urgent review needed.",
        "MEDIUM"   : f"Property crime detected: {predicted_class}. Flagged for review.",
        "LOW"      : f"Notable activity detected: {predicted_class}. Monitoring recommended.",
        "NORMAL"   : "No threat detected. Scene appears normal.",
    }

    return RiskResult(
        event_class  = predicted_class,
        risk_level   = risk_level,
        base_score   = base_score,
        final_score  = final_score,
        confidence   = confidence,
        alert        = alert,
        description  = descriptions[risk_level],
        color        = LEVEL_COLORS[risk_level],
    )


def evaluate_video(summary: dict) -> RiskResult:
    """
    Compute risk from a predict_video.py summary dict.

    Args:
        summary : dict with keys 'final_prediction' and 'confidence'

    Returns:
        RiskResult
    """
    return compute_risk(
        predicted_class = summary["final_prediction"],
        confidence      = summary["confidence"],
    )


def print_risk(result: RiskResult) -> None:
    """Pretty-print a RiskResult to console."""
    print("\n── Risk Assessment ────────────────────────────────────")
    print(f"  {result.color} Risk Level  : {result.risk_level}")
    print(f"  Event        : {result.event_class}")
    print(f"  Base Score   : {result.base_score:.2f}")
    print(f"  Confidence   : {result.confidence*100:.1f}%")
    print(f"  Final Score  : {result.final_score:.4f}")
    print(f"  Alert        : {'YES ⚠' if result.alert else 'NO'}")
    print(f"  Description  : {result.description}")
    print("───────────────────────────────────────────────────────\n")


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("Shooting",      0.91),
        ("Fighting",      0.76),
        ("Shoplifting",   0.65),
        ("RoadAccidents", 0.80),
        ("NormalVideos",  0.95),
    ]

    for cls, conf in test_cases:
        result = compute_risk(cls, conf)
        print_risk(result)