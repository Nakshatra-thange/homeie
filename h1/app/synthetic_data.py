"""
app/synthetic_data.py

Generates realistic, messy, multi-source rental listing data for Gurgaon —
standing in for what Homeie would actually scrape/collect from:
  - PLATFORM      : 99acres, MagicBricks, Housing.com
  - HIDDEN_GROUP   : WhatsApp/Facebook broker circles that never platform
  - OFF_MARKET     : units only reachable via direct broker/owner relationships

The key design idea: we first generate a set of "ground truth" PHYSICAL
UNITS (real homes), then simulate how each one surfaces across sources —
sometimes 3 times with three different brokers and three slightly different
prices/descriptions, sometimes once in a WhatsApp group and never online,
sometimes not at all until Homeie's own network finds it.

This reproduces the exact funnel Homeie advertises on their landing page:
    120 listed on platforms -> 36 unique after dedup
    172 posted in hidden groups
    210 never posted anywhere

We keep a `_true_unit_id` on every generated Listing purely so Part 2's
dedup pipeline can be scored against a real answer key (precision/recall),
which most take-home dedup demos skip and which is exactly what makes this
one credible.
"""

import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from app.models import Address, Facing, Furnishing, Listing, SourceType

random.seed(42)  # deterministic output — reproducible runs, reviewable diffs

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SOCIETIES = [
    ("M3M Heights", "65"), ("Ireo Skyon", "60"), ("Birla Navya", "63A"),
    ("DLF Regal Gardens", "90"), ("Emaar Digi Homes", "62"),
    ("Central Park Flower Valley", "33"), ("Vatika India Next", "82"),
    ("Godrej Habitat", "78"), ("Sobha City", "108"), ("Ambience Creacions", "22"),
    ("Adani Oyster Grande", "102"), ("Tulip Violet", "69"), ("Bestech Park View", "48"),
    ("ATS Kocoon", "109"), ("Experion Windchants", "112"),
]

AMENITIES_POOL = [
    "24x7 power backup", "covered parking", "clubhouse", "swimming pool", "gym",
    "children's play area", "jogging track", "pet friendly", "modular kitchen",
    "piped gas", "rainwater harvesting", "visitor parking", "CCTV surveillance",
]

PLATFORM_SOURCES = ["99acres", "MagicBricks", "Housing.com"]
HIDDEN_GROUP_SOURCES = [
    "Sohna Road Rentals (WA)", "GGN 3BHK Exchange (WA)",
    "DLF Phase Brokers Circle (WA)", "Golf Course Ext Owners Group (WA)",
]
OFF_MARKET_SOURCE = "Homeie Direct Network"

BROKER_FIRST = ["Rajesh", "Anita", "Vikram", "Priya", "Sanjay", "Neha", "Amit",
                "Sunita", "Deepak", "Kavita", "Manoj", "Ritu"]
BROKER_LAST = ["Sharma", "Verma", "Gupta", "Malhotra", "Chauhan", "Yadav",
               "Kapoor", "Singh", "Bansal", "Arora"]

DESC_TEMPLATES_PLATFORM = [
    "Spacious {bhk} BHK available in {society}, {facing}-facing, {sqft} sqft. Contact for viewing.",
    "{bhk}BHK for rent in {society}, Sector {sector}. {furnishing}, floor {floor}. Genuine owner listing.",
    "Well maintained {bhk} BHK, {society}. Ideal for families. {furnishing} unit, immediate possession.",
]
DESC_TEMPLATES_HIDDEN = [
    "{bhk}bhk {society} sec {sector} available!! {furnishing} floor {floor} call fast",
    "Owner shifting - {bhk} BHK {society} open for rent, brokers welcome 1 month",
    "Direct from owner: {society} {bhk}BHK, {sqft} sq ft approx, negotiable",
]


@dataclass
class PhysicalUnit:
    """Ground-truth home. Never exposed directly — only surfaces through
    the Listing instances generated to represent how it appears per source."""
    unit_id: str
    society: str
    sector: str
    tower: str
    bhk: int
    floor: int
    total_floors: int
    facing: Facing
    carpet_sqft: int
    base_rent: int
    maintenance: int
    club: int
    furnishing: Furnishing
    amenities: list[str]
    base_photo_hash: str


def _rand_phash() -> str:
    """A stand-in for a real perceptual hash (imagehash.phash on an actual
    photo). We use a 16-hex-char string (64 bits, same length as a real
    pHash) so Part 2 can do genuine Hamming-distance comparisons."""
    return "".join(random.choice("0123456789abcdef") for _ in range(16))


def _mutate_phash(h: str, max_flips: int = 3) -> str:
    """Simulate the same photo being re-compressed/cropped/watermarked when
    a different broker reposts it — a few bits change, but it's clearly
    'the same photo' under Hamming distance."""
    chars = list(h)
    n_flips = random.randint(1, max_flips)
    for _ in range(n_flips):
        i = random.randrange(len(chars))
        chars[i] = random.choice("0123456789abcdef")
    return "".join(chars)


def _generate_units(n: int) -> list[PhysicalUnit]:
    units = []
    for i in range(n):
        society, sector = random.choice(SOCIETIES)
        bhk = random.choice([2, 2, 3, 3, 3, 4])
        floor = random.randint(1, 28)
        base_rent = random.randint(35, 130) * 1000
        units.append(PhysicalUnit(
            unit_id=f"UNIT-{i:04d}",
            society=society,
            sector=sector,
            tower=f"T{random.randint(1, 9)}",
            bhk=bhk,
            floor=floor,
            total_floors=random.randint(floor, 35),
            facing=random.choice(list(Facing)),
            carpet_sqft=bhk * random.randint(430, 520),
            base_rent=base_rent,
            maintenance=round(base_rent * random.uniform(0.03, 0.07), -2).__int__(),
            club=random.choice([0, 1000, 1500, 2000, 2500]),
            furnishing=random.choice(list(Furnishing)),
            amenities=random.sample(AMENITIES_POOL, k=random.randint(3, 7)),
            base_photo_hash=_rand_phash(),
        ))
    return units


def _broker_name() -> str:
    return f"{random.choice(BROKER_FIRST)} {random.choice(BROKER_LAST)}"


def _make_listing(unit: PhysicalUnit, source_type: SourceType, source_name: str,
                   listing_seq: int, days_ago: int, active: bool,
                   price_jitter: float, sqft_jitter: int) -> Listing:
    templates = DESC_TEMPLATES_PLATFORM if source_type == SourceType.PLATFORM else DESC_TEMPLATES_HIDDEN
    desc = random.choice(templates).format(
        bhk=unit.bhk, society=unit.society, sector=unit.sector,
        facing=unit.facing.value, sqft=unit.carpet_sqft + sqft_jitter,
        furnishing=unit.furnishing.value, floor=unit.floor,
    )
    # brokers rarely list the full amenity set consistently — drop a few at random
    amenities = [a for a in unit.amenities if random.random() > 0.25]

    return Listing(
        listing_id=f"{source_name[:4].upper()}-{unit.unit_id}-{listing_seq}",
        source_type=source_type,
        source_name=source_name,
        posted_by=_broker_name(),
        posted_date=date.today() - timedelta(days=days_ago),
        is_active=active,
        address=Address(sector=unit.sector, society=unit.society, tower=unit.tower),
        bhk=unit.bhk,
        floor=unit.floor,
        total_floors=unit.total_floors,
        facing=unit.facing,
        carpet_sqft=unit.carpet_sqft + sqft_jitter,
        rent=int(unit.base_rent * price_jitter),
        maintenance=unit.maintenance,
        club=unit.club,
        furnishing=unit.furnishing,
        amenities=amenities,
        photo_hashes=[_mutate_phash(unit.base_photo_hash) for _ in range(random.randint(1, 4))],
        description=desc,
        _true_unit_id=unit.unit_id,
    )


def generate_all(n_units: int = 130) -> dict[str, list[Listing]]:
    """Returns the three raw source feeds, unmerged and undeduped — exactly
    what Part 2 will have to reconcile.

    Each physical unit is assigned a "reach tier" — the best channel it's
    discoverable through — with weights roughly matching Homeie's own funnel
    (a small platform-visible slice, a larger hidden-group slice, and the
    largest slice never posted anywhere). Units in the PLATFORM tier get
    multiple, independently-worded listings across brokers/platforms — that
    repost noise is exactly what Part 2's dedup pipeline has to collapse.
    """
    units = _generate_units(n_units)
    platform_listings, hidden_listings, off_market_listings = [], [], []

    for unit in units:
        tier = random.choices(
            ["platform", "hidden_group", "off_market"],
            weights=[0.23, 0.35, 0.42],
            k=1,
        )[0]

        if tier == "platform":
            chosen_platforms = random.sample(PLATFORM_SOURCES, k=random.randint(1, 3))
            for seq, platform in enumerate(chosen_platforms, start=1):
                platform_listings.append(_make_listing(
                    unit, SourceType.PLATFORM, platform, seq,
                    days_ago=random.randint(0, 45),
                    active=random.random() > 0.15,  # ~15% are stale/dead posts
                    price_jitter=random.uniform(1.0, 1.08),  # broker markup varies
                    sqft_jitter=random.randint(-15, 15),
                ))
            # platform-tier units often ALSO leak into a hidden group
            if random.random() < 0.3:
                group = random.choice(HIDDEN_GROUP_SOURCES)
                hidden_listings.append(_make_listing(
                    unit, SourceType.HIDDEN_GROUP, group, 1,
                    days_ago=random.randint(0, 20), active=True,
                    price_jitter=random.uniform(0.97, 1.05), sqft_jitter=random.randint(-25, 25),
                ))

        elif tier == "hidden_group":
            chosen_groups = random.sample(HIDDEN_GROUP_SOURCES, k=random.randint(1, 2))
            for seq, group in enumerate(chosen_groups, start=1):
                hidden_listings.append(_make_listing(
                    unit, SourceType.HIDDEN_GROUP, group, seq,
                    days_ago=random.randint(0, 20),
                    active=random.random() > 0.05,
                    price_jitter=random.uniform(0.97, 1.05),
                    sqft_jitter=random.randint(-25, 25),
                ))

        else:  # off_market — Homeie's own network is the only way to reach it
            off_market_listings.append(_make_listing(
                unit, SourceType.OFF_MARKET, OFF_MARKET_SOURCE, 1,
                days_ago=random.randint(0, 10),
                active=True,
                price_jitter=1.0,
                sqft_jitter=0,
            ))

    return {
        "platform": platform_listings,
        "hidden_group": hidden_listings,
        "off_market": off_market_listings,
        "_ground_truth_unit_count": len(units),
    }


def save_to_disk(feeds: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for key in ("platform", "hidden_group", "off_market"):
        path = DATA_DIR / f"source_{key}.json"
        payload = [listing.to_dict() for listing in feeds[key]]
        path.write_text(json.dumps(payload, indent=2))
        print(f"  wrote {path.relative_to(DATA_DIR.parent.parent)}  ({len(payload)} listings)")


if __name__ == "__main__":
    feeds = generate_all()
    save_to_disk(feeds)

    n_platform = len(feeds["platform"])
    n_hidden = len(feeds["hidden_group"])
    n_off_market = len(feeds["off_market"])
    n_active_platform = sum(1 for l in feeds["platform"] if l.is_active)
    unique_platform_units = len({l._true_unit_id for l in feeds["platform"]})

    print("\n--- funnel summary (mirrors the Homeie landing page stats) ---")
    print(f"  Listed on platforms:            {n_platform}")
    print(f"  Left after removing dead posts: {n_active_platform}")
    print(f"  Unique physical units (platform only): {unique_platform_units}")
    print(f"  Posted live in hidden groups:    {n_hidden}")
    print(f"  Never posted anywhere at all:    {n_off_market}")
    print(f"  Ground-truth physical units:     {feeds['_ground_truth_unit_count']}")