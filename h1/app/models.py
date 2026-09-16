"""
app/models.py

Core data models for the Homeie Dedup + HomeScore engine.

Design note: `listing_id` is only unique WITHIN a single source. Two Listing
objects from different sources (or even the same source, reposted) can refer
to the exact same physical home. Resolving that — deciding which listings are
"the same unit" — is exactly the problem Part 2 (dedup) solves. Part 1 only
defines the schema and produces realistic, messy, multi-source raw data.
"""

from dataclasses import dataclass, field, asdict
from datetime import date
from enum import Enum
from typing import Optional


class SourceType(str, Enum):
    PLATFORM = "platform"          # 99acres / MagicBricks / Housing.com style listing sites
    HIDDEN_GROUP = "hidden_group"  # WhatsApp / Facebook broker groups, never platformed
    OFF_MARKET = "off_market"      # never posted anywhere; reached via direct broker/owner relationships


class Facing(str, Enum):
    N = "N"
    S = "S"
    E = "E"
    W = "W"
    NE = "NE"
    NW = "NW"
    SE = "SE"
    SW = "SW"


class Furnishing(str, Enum):
    UNFURNISHED = "unfurnished"
    SEMI_FURNISHED = "semi-furnished"
    FULLY_FURNISHED = "fully-furnished"


@dataclass
class Address:
    sector: str
    society: str
    tower: Optional[str] = None
    city: str = "Gurgaon"

    def normalized(self) -> str:
        """A loose, human-ish normalized string. Deliberately NOT strict —
        Part 2's fuzzy matcher is what has to handle real-world messiness
        (typos, 'Sec' vs 'Sector', tower omitted, etc.), not this method."""
        parts = [self.society, self.tower or "", f"Sector {self.sector}", self.city]
        return " ".join(p.strip() for p in parts if p and p.strip()).lower()


@dataclass
class Listing:
    listing_id: str
    source_type: SourceType
    source_name: str        # e.g. "99acres", "Sohna Road Rentals (WA)", "Homeie Direct Network"
    posted_by: str           # broker or owner display name as shown on that source
    posted_date: date
    is_active: bool          # False = stale/dead post still sitting on the platform

    address: Address
    bhk: int
    floor: int
    total_floors: int
    facing: Facing
    carpet_sqft: int

    rent: int
    maintenance: int
    club: int

    furnishing: Furnishing
    amenities: list[str] = field(default_factory=list)
    photo_hashes: list[str] = field(default_factory=list)  # simulated perceptual hashes, see synthetic_data.py
    description: str = ""

    # A hidden field ONLY used for validating the dedup pipeline in Part 2.
    # Real ingestion would never have this — it's ground truth we generate
    # ourselves so we can measure dedup precision/recall against an answer key.
    _true_unit_id: str = ""

    @property
    def total_monthly_cost(self) -> int:
        return self.rent + self.maintenance + self.club

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_type"] = self.source_type.value
        d["facing"] = self.facing.value
        d["furnishing"] = self.furnishing.value
        d["posted_date"] = self.posted_date.isoformat()
        return d