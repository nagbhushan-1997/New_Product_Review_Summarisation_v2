"""
Stage 3 (v2): build the products.jsonl file that app.py reads.

Joins image-bearing reviews from Stage 1-2 against raw_meta_Electronics on parent_asin,
computes a lightweight contradiction_score per review (for the Contradiction-Rich Queue),
and writes one product per line.
"""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re

MISMATCH_PATTERN = re.compile(
    r"\b(broken|broke|crack(?:ed|s)?|scratch(?:ed|es)?|damaged|dent(?:ed)?|"
    r"wrong|different|missing|fake|counterfeit|cheap|flimsy|peeling|torn|"
    r"not as pictured|not as described|doesn['’]?t fit|didn['’]?t fit|"
    r"stopped working|fell apart|defective|poor quality|misleading|color is off)\b",
    re.IGNORECASE,
)


def compute_contradiction_score(r: dict) -> float:
    score = 0.0
    rating = float(r.get("rating") or 5.0)
    if rating <= 2.0:
        score += 3.0
    elif rating <= 3.0:
        score += 2.0
    text = f"{r.get('title') or ''} {r.get('text') or ''}"
    hits = len(MISMATCH_PATTERN.findall(text))
    score += min(hits, 4) * 1.25
    if rating >= 4.0 and hits >= 1:
        score += 2.0
    if len(r.get("image_urls") or []) >= 2:
        score += 0.5
    return round(score, 2)


def iter_meta(meta_mode: str, local_meta_file: Path | None):
    if meta_mode == "stream":
        from datasets import load_dataset

        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_meta_Electronics",
            split="full",
            streaming=True,
        )
        for rec in ds:
            yield rec
    else:
        opener = gzip.open if local_meta_file.suffix == ".gz" else open
        with opener(local_meta_file, "rt", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    yield json.loads(line)


def catalog_images(raw, limit=3):
    if not isinstance(raw, dict):
        return []
    for key in ("hi_res", "large", "thumb"):
        vals = raw.get(key) or []
        urls = [u for u in vals if u and u != "None"]
        if urls:
            return urls[:limit]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-reviews", required=True, help="Path to electronics_image_reviews.jsonl")
    parser.add_argument("--out", default="products.jsonl", help="Output products.jsonl path")
    parser.add_argument("--meta-mode", choices=["stream", "local"], default="stream")
    parser.add_argument("--local-meta", default=None, help="Path to meta_Electronics.jsonl(.gz)")
    parser.add_argument("--min-reviews", type=int, default=5)
    parser.add_argument("--max-reviews", type=int, default=30)
    parser.add_argument("--max-products", type=int, default=20000)
    args = parser.parse_args()

    by_product = defaultdict(list)
    with Path(args.image_reviews).open("r", encoding="utf-8") as fp:
        for line in fp:
            r = json.loads(line)
            pa = r.get("parent_asin")
            if pa:
                r["contradiction_score"] = compute_contradiction_score(r)
                by_product[pa].append(r)

    eligible = {}
    for pa, revs in by_product.items():
        if len(revs) < args.min_reviews:
            continue
        revs.sort(
            key=lambda r: (r.get("contradiction_score", 0), r.get("n_images", 0), r.get("helpful_vote") or 0),
            reverse=True,
        )
        eligible[pa] = revs[: args.max_reviews]

    print(f"{len(by_product):,} products seen, {len(eligible):,} with >= {args.min_reviews} image reviews")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    wanted = set(eligible)
    local_meta = Path(args.local_meta) if args.local_meta else None

    with out_path.open("w", encoding="utf-8") as out:
        for i, m in enumerate(iter_meta(args.meta_mode, local_meta), 1):
            if i % 200_000 == 0:
                print(f"  scanned {i:,} meta records, matched {written:,}", flush=True)
            pa = m.get("parent_asin")
            if pa not in wanted:
                continue
            desc = m.get("description") or []
            feats = m.get("features") or []
            product = {
                "parent_asin": pa,
                "title": m.get("title") or "(untitled)",
                "store": m.get("store"),
                "price": m.get("price"),
                "average_rating": m.get("average_rating"),
                "rating_number": m.get("rating_number"),
                "main_category": m.get("main_category"),
                "categories": m.get("categories") or [],
                "features": feats[:8],
                "description": " ".join(desc)[:1200] if desc else "",
                "catalog_images": catalog_images(m.get("images")),
                "reviews": eligible[pa],
            }
            out.write(json.dumps(product, ensure_ascii=False) + "\n")
            written += 1
            wanted.discard(pa)
            if written >= args.max_products or not wanted:
                break

    print(f"Wrote {written:,} products to {out_path}")


if __name__ == "__main__":
    main()
