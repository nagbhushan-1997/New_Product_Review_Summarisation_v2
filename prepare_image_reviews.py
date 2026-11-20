"""
Stage 1-2 (v2): filter raw Amazon Reviews '23 Electronics reviews down to those
with at least one reviewer-uploaded photo.

Source: McAuley-Lab/Amazon-Reviews-2023 (config: raw_review_Electronics).
"""

import argparse
import gzip
import json
from pathlib import Path


def iter_reviews(mode: str, local_file: Path | None):
    if mode == "stream":
        from datasets import load_dataset

        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_review_Electronics",
            split="full",
            streaming=True,
            trust_remote_code=True,
        )
        for rec in ds:
            yield rec
    else:
        opener = gzip.open if local_file.suffix == ".gz" else open
        with opener(local_file, "rt", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    yield json.loads(line)


def extract_image_urls(images):
    urls = []
    for img in images or []:
        if not isinstance(img, dict):
            continue
        url = (
            img.get("large_image_url")
            or img.get("medium_image_url")
            or img.get("small_image_url")
        )
        if url:
            urls.append(url)
    return urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="electronics_image_reviews.jsonl")
    parser.add_argument("--mode", choices=["stream", "local"], default="stream")
    parser.add_argument("--local-file", default=None)
    parser.add_argument("--progress-every", type=int, default=200_000)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    local_file = Path(args.local_file) if args.local_file else None

    seen = 0
    kept = 0
    with out_path.open("w", encoding="utf-8") as out:
        for rec in iter_reviews(args.mode, local_file):
            seen += 1
            if seen % args.progress_every == 0:
                print(f"  scanned {seen:,} reviews, kept {kept:,}", flush=True)
            urls = extract_image_urls(rec.get("images"))
            if not urls or not rec.get("parent_asin"):
                continue
            row = {
                "parent_asin": rec.get("parent_asin"),
                "asin": rec.get("asin"),
                "user_id": rec.get("user_id"),
                "timestamp": rec.get("timestamp"),
                "rating": rec.get("rating"),
                "title": rec.get("title"),
                "text": rec.get("text"),
                "verified_purchase": rec.get("verified_purchase"),
                "helpful_vote": rec.get("helpful_vote"),
                "image_urls": urls,
                "n_images": len(urls),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Scanned {seen:,} reviews, wrote {kept:,} with photos to {out_path}")


if __name__ == "__main__":
    main()
