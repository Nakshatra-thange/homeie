"""
app/dedup.py

The dedup pipeline: takes raw, undeduped listings from multiple sources and
collapses reposts of the same physical unit into a single canonical record.

Approach
--------
1. Hard filters (cheap, must-match): same BHK, same sector, same floor.
   These rarely get misreported even by sloppy brokers, so requiring exact
   agreement on them is a strong precision guardrail and also acts as a
   *blocking key* — we never even compare listings outside their own
   (bhk, sector, floor) bucket, which keeps this roughly O(n) in practice
   instead of O(n^2) over the whole dataset.

2. Weighted similarity (soft signals, the things brokers routinely fudge):
     - address text similarity (society/tower spelling, "Sec" vs "Sector")
     - price similarity (broker markup varies listing to listing)
     - carpet area similarity (brokers round/estimate sqft)
     - photo hash similarity (Hamming distance on simulated pHashes —
       the same photo re-uploaded by a different broker still looks like
       the same photo bit-wise, even after recompression)

3. Any pair scoring above `MATCH_THRESHOLD` is unioned together via a
   union-find structure, so matches are transitive: if listing A matches B
   and B matches C, all three end up in one cluster even if A and C were
   never compared directly (this matters when 3 different brokers describe
   the same unit slightly differently — A and C might not be similar enough
   on their own).

4. Each resulting cluster is merged into one canonical `UniqueUnit` —
   picking representative values and keeping full provenance of every
   source listing that contributed.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from app.models import Listing

# ---------------------------------------------------------------------------
# Tunable weights. These were tuned by running evaluate_clusters() against
# the synthetic ground truth in Part 1 and adjusting until pairwise
# precision and recall were both comfortably high (see README for the run).
# ---------------------------------------------------------------------------
WEIGHT_ADDRESS = 0.40
WEIGHT_PRICE = 0.25
WEIGHT_SQFT = 0.15
WEIGHT_PHOTO = 0.20
MATCH_THRESHOLD = 0.70

# If neither listing has a usable photo hash, we redistribute WEIGHT_PHOTO
# across the remaining signals rather than silently scoring it as 0.


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def address_similarity(a: Listing, b: Listing) -> float:
    """Blend of whole-string similarity and token overlap, so it tolerates
    both minor spelling drift ('Ireo Skyon' vs 'Ireo Skyon Twr') and
    reordering ('T4 M3M Heights' vs 'M3M Heights T4')."""
    sa, sb = a.address.normalized(), b.address.normalized()
    seq_ratio = difflib.SequenceMatcher(None, sa, sb).ratio()

    tokens_a, tokens_b = set(sa.split()), set(sb.split())
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if (tokens_a | tokens_b) else 0.0

    return 0.5 * seq_ratio + 0.5 * jaccard


def price_similarity(a: Listing, b: Listing) -> float:
    ca, cb = a.total_monthly_cost, b.total_monthly_cost
    if max(ca, cb) == 0:
        return 1.0
    diff_ratio = abs(ca - cb) / max(ca, cb)
    return _clamp01(1 - diff_ratio / 0.15)  # tolerate up to ~15% broker markup spread


def sqft_similarity(a: Listing, b: Listing) -> float:
    if max(a.carpet_sqft, b.carpet_sqft) == 0:
        return 1.0
    diff_ratio = abs(a.carpet_sqft - b.carpet_sqft) / max(a.carpet_sqft, b.carpet_sqft)
    return _clamp01(1 - diff_ratio / 0.08)  # tolerate up to ~8% sqft estimation error


def _hamming(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def photo_similarity(a: Listing, b: Listing) -> float | None:
    """Returns None (not just 0) when there's nothing to compare, so the
    caller can redistribute that weight instead of unfairly penalizing
    listings with no photos."""
    if not a.photo_hashes or not b.photo_hashes:
        return None
    min_dist = min(_hamming(h1, h2) for h1 in a.photo_hashes for h2 in b.photo_hashes)
    # hashes are 64 bits; true reposts differ by a handful of bits from
    # recompression, unrelated photos differ by ~32 bits on average
    return _clamp01(1 - min_dist / 16)


def passes_hard_filters(a: Listing, b: Listing) -> bool:
    return (
        a.bhk == b.bhk
        and a.address.sector.strip().lower() == b.address.sector.strip().lower()
        and a.floor == b.floor
    )


def similarity(a: Listing, b: Listing) -> float:
    if not passes_hard_filters(a, b):
        return 0.0

    addr = address_similarity(a, b)
    price = price_similarity(a, b)
    sqft = sqft_similarity(a, b)
    photo = photo_similarity(a, b)

    if photo is None:
        total_weight = WEIGHT_ADDRESS + WEIGHT_PRICE + WEIGHT_SQFT
        score = (WEIGHT_ADDRESS * addr + WEIGHT_PRICE * price + WEIGHT_SQFT * sqft) / total_weight
    else:
        score = (
            WEIGHT_ADDRESS * addr + WEIGHT_PRICE * price + WEIGHT_SQFT * sqft + WEIGHT_PHOTO * photo
        )

    return score


# ---------------------------------------------------------------------------
# Union-Find (disjoint set) — makes matches transitive across pairs
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def cluster_listings(listings: list[Listing], threshold: float = MATCH_THRESHOLD) -> list[list[Listing]]:
    """Blocks by (bhk, sector, floor) to avoid comparing every listing to
    every other listing, then unions any pair scoring above threshold
    within each block."""
    n = len(listings)
    uf = UnionFind(n)

    blocks: dict[tuple, list[int]] = {}
    for i, l in enumerate(listings):
        key = (l.bhk, l.address.sector.strip().lower(), l.floor)
        blocks.setdefault(key, []).append(i)

    comparisons = 0
    for indices in blocks.values():
        for i_pos in range(len(indices)):
            for j_pos in range(i_pos + 1, len(indices)):
                i, j = indices[i_pos], indices[j_pos]
                comparisons += 1
                if similarity(listings[i], listings[j]) >= threshold:
                    uf.union(i, j)

    groups: dict[int, list[Listing]] = {}
    for i, l in enumerate(listings):
        root = uf.find(i)
        groups.setdefault(root, []).append(l)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Merging a cluster into one canonical record
# ---------------------------------------------------------------------------
@dataclass
class SourceRef:
    source_type: str
    source_name: str
    listing_id: str
    rent: int
    total_monthly_cost: int
    is_active: bool
    posted_date: str


@dataclass
class UniqueUnit:
    unit_id: str
    society: str
    sector: str
    tower: str | None
    bhk: int
    floor: int
    total_floors: int
    facing: str
    carpet_sqft: int

    rent_min: int
    rent_max: int
    rent_representative: int  # median across contributing listings
    maintenance: int
    club: int

    furnishing: str
    amenities: list[str]
    is_active_anywhere: bool
    is_off_market_only: bool
    num_duplicate_listings: int
    sources: list[SourceRef] = field(default_factory=list)
    description_sample: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["sources"] = [s.__dict__ for s in self.sources]
        return d


def _median(values: list[int]) -> int:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) // 2


def merge_cluster(cluster: list[Listing], unit_index: int) -> UniqueUnit:
    # representative = the listing with the most complete amenities list,
    # a reasonable proxy for "most carefully written" post
    representative = max(cluster, key=lambda l: len(l.amenities))

    rents = [l.rent for l in cluster]
    all_amenities = sorted({a for l in cluster for a in l.amenities})
    source_types = {l.source_type.value for l in cluster}

    return UniqueUnit(
        unit_id=f"HOMEIE-{unit_index:04d}",
        society=representative.address.society,
        sector=representative.address.sector,
        tower=representative.address.tower,
        bhk=representative.bhk,
        floor=representative.floor,
        total_floors=representative.total_floors,
        facing=representative.facing.value,
        carpet_sqft=round(sum(l.carpet_sqft for l in cluster) / len(cluster)),
        rent_min=min(rents),
        rent_max=max(rents),
        rent_representative=_median(rents),
        maintenance=representative.maintenance,
        club=representative.club,
        furnishing=representative.furnishing.value,
        amenities=all_amenities,
        is_active_anywhere=any(l.is_active for l in cluster),
        is_off_market_only=(source_types == {"off_market"}),
        num_duplicate_listings=len(cluster),
        sources=[
            SourceRef(
                source_type=l.source_type.value,
                source_name=l.source_name,
                listing_id=l.listing_id,
                rent=l.rent,
                total_monthly_cost=l.total_monthly_cost,
                is_active=l.is_active,
                posted_date=l.posted_date.isoformat(),
            )
            for l in cluster
        ],
        description_sample=representative.description,
    )


def merge_all(clusters: list[list[Listing]]) -> list[UniqueUnit]:
    return [merge_cluster(c, i) for i, c in enumerate(clusters, start=1)]


# ---------------------------------------------------------------------------
# Evaluation against the synthetic ground truth (_true_unit_id)
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    predicted_clusters: int
    true_unique_units: int
    exact_cluster_matches: int  # predicted clusters that exactly equal a true unit's listing set


def evaluate_clusters(clusters: list[list[Listing]], all_listings: list[Listing]) -> EvalResult:
    """Standard entity-resolution evaluation: look at every PAIR of listings.
    Precision = of the pairs we put in the same cluster, how many are truly
    the same unit. Recall = of the pairs that are truly the same unit, how
    many did we actually cluster together."""
    from itertools import combinations

    true_pairs = set()
    by_true_id: dict[str, list[int]] = {}
    for i, l in enumerate(all_listings):
        by_true_id.setdefault(l._true_unit_id, []).append(i)
    for indices in by_true_id.values():
        for i, j in combinations(sorted(indices), 2):
            true_pairs.add((i, j))

    id_to_index = {id(l): i for i, l in enumerate(all_listings)}
    predicted_pairs = set()
    for cluster in clusters:
        indices = sorted(id_to_index[id(l)] for l in cluster)
        for i, j in combinations(indices, 2):
            predicted_pairs.add((i, j))

    tp = len(true_pairs & predicted_pairs)
    fp = len(predicted_pairs - true_pairs)
    fn = len(true_pairs - predicted_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    true_unit_ids = {l._true_unit_id for l in all_listings}
    exact_matches = 0
    for cluster in clusters:
        true_ids_in_cluster = {l._true_unit_id for l in cluster}
        if len(true_ids_in_cluster) == 1:
            tid = next(iter(true_ids_in_cluster))
            total_listings_for_tid = sum(1 for l in all_listings if l._true_unit_id == tid)
            if len(cluster) == total_listings_for_tid:
                exact_matches += 1

    return EvalResult(
        pairwise_precision=precision,
        pairwise_recall=recall,
        pairwise_f1=f1,
        predicted_clusters=len(clusters),
        true_unique_units=len(true_unit_ids),
        exact_cluster_matches=exact_matches,
    )