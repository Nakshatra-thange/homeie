"""
app/scoring.py

The HomeScore engine: given a UserProfile and an enriched UniqueUnit,
produces a 0-100 overall score built from seven transparent 0-100
sub-scores (Sunlight, Noise, Water, Condition, Society, Neighborhood,
Commute) — mirroring the exact bars on Homeie's own landing page, plus
commute since their own demo interview leads with "where's the office?".

Every sub-score is a pure function of (unit, enrichment, profile) so the
breakdown shown to the user is the literal thing that produced the ranking
— not a black box justified after the fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.dedup import UniqueUnit
from app.enrichment import Enrichment
from app.profile import UserProfile

FACING_LIGHT_QUALITY = {
    # South and East-facing catch the most usable daylight in the northern
    # hemisphere; North is weakest; combinations sit in between.
    "S": 1.00, "SE": 0.95, "E": 0.90, "SW": 0.80,
    "NE": 0.65, "W": 0.60, "NW": 0.45, "N": 0.35,
}


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Individual sub-scores — each 0..100
# ---------------------------------------------------------------------------

def sunlight_score(unit: UniqueUnit, enr: Enrichment) -> float:
    facing_quality = FACING_LIGHT_QUALITY.get(unit.facing, 0.5)
    # higher floors catch more direct light (fewer neighboring buildings
    # blocking the window), with diminishing returns past ~15 floors
    floor_bonus = min(unit.floor, 15) / 15
    score = facing_quality * 75 + floor_bonus * 25
    return _clamp(score)


def noise_score(unit: UniqueUnit, enr: Enrichment) -> float:
    # higher floors are quieter (further from street-level traffic noise)
    floor_relief = min(unit.floor, 12) / 12
    base = (1 - enr.road_noise_index) * 70 + floor_relief * 30
    return _clamp(base)


def water_score(unit: UniqueUnit, enr: Enrichment) -> float:
    return _clamp(enr.water_quality_index * 100)


def condition_score(unit: UniqueUnit, enr: Enrichment) -> float:
    furnishing_bonus = {"fully-furnished": 20, "semi-furnished": 10, "unfurnished": 0}
    age_penalty = min(enr.building_age_years, 20) * 2  # older buildings score lower
    base = 80 - age_penalty + furnishing_bonus.get(unit.furnishing, 0)
    return _clamp(base)


def society_score(unit: UniqueUnit, enr: Enrichment) -> float:
    amenity_bonus = min(len(unit.amenities), 8) * 3.5  # up to +28
    return _clamp(enr.society_reputation_index * 72 + amenity_bonus)


def neighborhood_score(unit: UniqueUnit, enr: Enrichment) -> float:
    return _clamp(enr.neighborhood_index * 100)


def commute_score(enr: Enrichment, profile: UserProfile) -> float | None:
    if profile.office_lat is None or profile.office_lon is None:
        return None
    dist_km = _haversine_km(enr.lat, enr.lon, profile.office_lat, profile.office_lon)
    # 0 km -> 100, ~20km+ in Gurgaon traffic -> near 0, roughly linear with a floor
    score = 100 - dist_km * 4.5
    return _clamp(score), dist_km  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Combined result
# ---------------------------------------------------------------------------
@dataclass
class ScoreBreakdown:
    unit_id: str
    home_score: int
    sub_scores: dict[str, float]
    commute_km: float | None
    passes_hard_filters: bool
    filter_reasons: list[str] = field(default_factory=list)
    trade_offs: list[str] = field(default_factory=list)


def check_hard_filters(unit: UniqueUnit, profile: UserProfile) -> list[str]:
    reasons = []
    if profile.no_ground_floor and unit.floor <= 0:
        reasons.append("Ground floor — you asked to exclude these")
    if profile.budget_max is not None and unit.rent_representative > profile.budget_max:
        over = unit.rent_representative - profile.budget_max
        reasons.append(f"Rs.{over:,} over your Rs.{profile.budget_max:,} budget")
    if profile.min_bhk is not None and unit.bhk < profile.min_bhk:
        reasons.append(f"{unit.bhk}BHK is below your minimum of {profile.min_bhk}BHK")
    if profile.max_bhk is not None and unit.bhk > profile.max_bhk:
        reasons.append(f"{unit.bhk}BHK is above your maximum of {profile.max_bhk}BHK")
    return reasons


def _trade_offs(sub_scores: dict[str, float], commute_km: float | None) -> list[str]:
    """Rule-based, human-readable callouts — the same spirit as Homeie's
    'trade-offs worth knowing before you visit', generated from the actual
    sub-scores rather than free-text generation."""
    notes = []
    STRONG, WEAK = 75, 45

    label_map = {
        "sunlight": "Sunlight", "noise": "Quiet", "water": "Water quality",
        "condition": "Condition", "society": "Society upkeep", "neighborhood": "Neighborhood",
        "commute": "Commute",
    }
    strengths = [label_map[k] for k, v in sub_scores.items() if v >= STRONG]
    weaknesses = [label_map[k] for k, v in sub_scores.items() if v <= WEAK]

    if strengths:
        notes.append(f"Strong on: {', '.join(strengths)}.")
    if weaknesses:
        notes.append(f"Worth checking in person: {', '.join(weaknesses)}.")
    if commute_km is not None:
        notes.append(f"~{commute_km:.1f} km from your office.")
    return notes


def score_unit(unit: UniqueUnit, enr: Enrichment, profile: UserProfile) -> ScoreBreakdown:
    filter_reasons = check_hard_filters(unit, profile)

    raw_sub = {
        "sunlight": sunlight_score(unit, enr),
        "noise": noise_score(unit, enr),
        "water": water_score(unit, enr),
        "condition": condition_score(unit, enr),
        "society": society_score(unit, enr),
        "neighborhood": neighborhood_score(unit, enr),
    }
    commute_km = None
    commute_result = commute_score(enr, profile)
    if commute_result is not None:
        c_score, commute_km = commute_result
        raw_sub["commute"] = c_score

    weights = profile.weights()
    if "commute" not in raw_sub:
        weights.pop("commute", None)
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}

    overall = sum(raw_sub[k] * weights[k] for k in raw_sub)

    # hard-filter failures don't get silently hidden — they're still scored
    # (useful for "close but not quite" cases) but heavily penalized
    if filter_reasons:
        overall *= 0.5

    return ScoreBreakdown(
        unit_id=unit.unit_id,
        home_score=round(_clamp(overall)),
        sub_scores={k: round(v, 1) for k, v in raw_sub.items()},
        commute_km=round(commute_km, 2) if commute_km is not None else None,
        passes_hard_filters=not filter_reasons,
        filter_reasons=filter_reasons,
        trade_offs=_trade_offs(raw_sub, commute_km),
    )


def rank_units(
    units: list[UniqueUnit],
    enrichments: dict[str, Enrichment],
    profile: UserProfile,
    include_filtered_out: bool = False,
) -> list[ScoreBreakdown]:
    scored = [score_unit(u, enrichments[u.unit_id], profile) for u in units]
    if not include_filtered_out:
        scored = [s for s in scored if s.passes_hard_filters]
    return sorted(scored, key=lambda s: -s.home_score)