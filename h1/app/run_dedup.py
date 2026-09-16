"""
app/run_dedup.py

Loads the three raw source feeds written by synthetic_data.py, runs the
dedup pipeline across ALL of them combined (a unit posted on 99acres AND in
a WhatsApp group must still collapse into one record), evaluates the result
against the ground-truth `_true_unit_id` labels, and writes the deduped,
canonical unit list to disk for Part 3 (scoring) to consume.
"""

import json
from datetime import date
from pathlib import Path

from app.dedup import cluster_listings, evaluate_clusters, merge_all
from app.models import Address, Facing, Furnishing, Listing, SourceType

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_listings(path: Path) -> list[Listing]:
    raw = json.loads(path.read_text())
    listings = []
    for d in raw:
        listings.append(Listing(
            listing_id=d["listing_id"],
            source_type=SourceType(d["source_type"]),
            source_name=d["source_name"],
            posted_by=d["posted_by"],
            posted_date=date.fromisoformat(d["posted_date"]),
            is_active=d["is_active"],
            address=Address(**d["address"]),
            bhk=d["bhk"],
            floor=d["floor"],
            total_floors=d["total_floors"],
            facing=Facing(d["facing"]),
            carpet_sqft=d["carpet_sqft"],
            rent=d["rent"],
            maintenance=d["maintenance"],
            club=d["club"],
            furnishing=Furnishing(d["furnishing"]),
            amenities=d["amenities"],
            photo_hashes=d["photo_hashes"],
            description=d["description"],
            _true_unit_id=d.get("_true_unit_id", ""),
        ))
    return listings


def load_all_sources() -> list[Listing]:
    listings = []
    for name in ("source_platform.json", "source_hidden_group.json", "source_off_market.json"):
        listings.extend(_load_listings(DATA_DIR / name))
    return listings


def run() -> None:
    all_listings = load_all_sources()
    print(f"Loaded {len(all_listings)} raw listings across all sources.\n")

    clusters = cluster_listings(all_listings)
    units = merge_all(clusters)

    result = evaluate_clusters(clusters, all_listings)

    print("--- dedup result ---")
    print(f"  Raw listings in:          {len(all_listings)}")
    print(f"  Unique units out:         {len(units)}")
    print(f"  True unique units:        {result.true_unique_units}")
    print(f"  Exact cluster matches:    {result.exact_cluster_matches} / {result.true_unique_units}")
    print()
    print("--- evaluation vs ground truth (pairwise) ---")
    print(f"  Precision: {result.pairwise_precision:.3f}   (of pairs we merged, how many were actually the same unit)")
    print(f"  Recall:    {result.pairwise_recall:.3f}   (of truly-same pairs, how many we actually merged)")
    print(f"  F1:        {result.pairwise_f1:.3f}")
    print()

    multi_listing_units = [u for u in units if u.num_duplicate_listings > 1]
    print(f"--- sample: units with the most duplicate listings collapsed ---")
    for u in sorted(multi_listing_units, key=lambda u: -u.num_duplicate_listings)[:3]:
        print(f"\n  {u.unit_id}: {u.society}, Sector {u.sector}, {u.bhk}BHK, floor {u.floor}")
        print(f"    Collapsed {u.num_duplicate_listings} listings -> rent range Rs.{u.rent_min:,}-{u.rent_max:,} "
              f"(representative Rs.{u.rent_representative:,})")
        for s in u.sources:
            print(f"      - [{s.source_type}] {s.source_name}  Rs.{s.rent:,}  "
                  f"{'active' if s.is_active else 'STALE'}  ({s.listing_id})")

    out_path = DATA_DIR / "deduped_units.json"
    out_path.write_text(json.dumps([u.to_dict() for u in units], indent=2))
    print(f"\nWrote {len(units)} deduped units -> {out_path.relative_to(DATA_DIR.parent.parent)}")


if __name__ == "__main__":
    run()