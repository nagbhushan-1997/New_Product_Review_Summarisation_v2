"""
Clean an existing cross-modal CSV export (such as Main_Data_Cross_Modal.csv),
compute inter-annotator agreement (exact agreement + Cohen's kappa), and
generate leak-free train / val / test splits grouped strictly by parent_asin.

Usage:
    python clean_and_split.py --input Main_Data_Cross_Modal.csv --outdir cleaned_splits
"""

import argparse
from collections import Counter
from pathlib import Path
import random

import pandas as pd

ANNOTATOR_MAP = {
    "nagbhushan": "Nagbhushan",
    "divya kempannavar": "Divya Kempannavar",
    "harshali": "Harshali",
    "girija": "Girija",
    "asmita": "Asmita",
    "asmta": "Asmita",
}

VALID_VERDICTS = {
    "Photo supports the review",
    "Photo contradicts the review",
    "Photo shows something the review never mentions",
    "Cannot tell from this photo",
}


def normalize_annotator(val: str) -> str:
    cleaned = " ".join(str(val or "").strip().split())
    return ANNOTATOR_MAP.get(cleaned.lower(), cleaned.title())


def compute_cohens_kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None, 0.0
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    labels = set(ca) | set(cb)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe >= 1.0:
        return 1.0, po
    return (po - pe) / (1.0 - pe), po


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw exported CSV")
    parser.add_argument("--outdir", default="cleaned_splits", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    raw_len = len(df)

    # 1. Remove accidental repeated header rows
    df = df[df["annotator"].astype(str).str.lower() != "annotator"].copy()
    df = df[df["verdict"].astype(str).str.lower() != "verdict"].copy()

    # 2. Normalize annotator names
    df["annotator"] = df["annotator"].apply(normalize_annotator)

    # 3. Drop Not judged & missing review_text
    df = df[df["verdict"].isin(VALID_VERDICTS)].copy()
    df = df[df["review_text"].fillna("").str.strip().ne("")].copy()

    # 4. Sort by updated_at and deduplicate (annotator, review_key), keeping latest
    df = df.sort_values("updated_at").drop_duplicates(
        subset=["annotator", "parent_asin", "review_key"], keep="last"
    )

    clean_path = outdir / "clean_annotations.csv"
    df.to_csv(clean_path, index=False)

    # 5. Compute IAA on multi-annotated reviews
    pairs = []
    for rk, grp in df.groupby("review_key"):
        if grp["annotator"].nunique() >= 2:
            vals = grp["verdict"].tolist()
            pairs.append((vals[0], vals[1]))
    kappa, exact = compute_cohens_kappa(pairs)

    # 6. Product-level (parent_asin) 70 / 15 / 15 split to prevent product leakage
    asins = sorted(df["parent_asin"].unique().tolist())
    rng = random.Random(args.seed)
    rng.shuffle(asins)
    n = len(asins)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train_asins = set(asins[:n_train])
    val_asins = set(asins[n_train : n_train + n_val])
    test_asins = set(asins[n_train + n_val :])

    df[df["parent_asin"].isin(train_asins)].to_csv(outdir / "train.csv", index=False)
    df[df["parent_asin"].isin(val_asins)].to_csv(outdir / "val.csv", index=False)
    df[df["parent_asin"].isin(test_asins)].to_csv(outdir / "test.csv", index=False)

    print(f"Raw rows: {raw_len} -> Clean rows: {len(df)}")
    print(f"Unique reviews: {df['review_key'].nunique()} | Unique products: {n}")
    print(f"Annotators: {dict(df['annotator'].value_counts())}")
    print(f"Verdicts: {dict(df['verdict'].value_counts())}")
    print(
        f"Multi-annotated reviews: {len(pairs)} | Exact agreement: {exact*100:.1f}% | "
        f"Cohen's kappa: {kappa:.3f}" if kappa is not None else "No multi-annotated pairs"
    )
    print(
        f"Leak-free parent_asin splits -> train: {len(train_asins)} products, "
        f"val: {len(val_asins)} products, test: {len(test_asins)} products"
    )


if __name__ == "__main__":
    main()
