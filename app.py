"""
Cross-modal review checker & multimodal product summary annotator (v2).

Upgrades over v1:
- Controlled annotator selector (eliminates casing/typo duplicates).
- Three annotation queues: Search/Random, Contradiction-Rich (1-3 star & defect/mismatch
  keywords to address class imbalance), and Blind IAA (double-annotation queue).
- Image informativeness tagging (Informative / Redundant / Noise) for visual gating research.
- Save validation guards (blocks 'Not judged', requires claim & contradiction notes).
- Tab 2: Product-level multimodal gold summary authoring (SQLite table: product_summaries).
- Tab 3: Live dataset audit, class-balance targets, and Cohen's kappa IAA metrics.

Run:  streamlit run app.py
"""

import csv
import io
import json
import os
import random
import re
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ------------------------------- config ------------------------------------
DATA_FILE = Path(__file__).parent / "products.jsonl"
DB_FILE = Path(__file__).parent / "annotations.db"


def _secret(key):
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO") or _secret("HF_DATASET_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN") or _secret("HF_TOKEN")


def ensure_products_file():
    if DATA_FILE.exists() or not HF_DATASET_REPO:
        return
    from huggingface_hub import hf_hub_download

    with st.spinner("Downloading product catalog (first run only)..."):
        downloaded = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename="products.jsonl",
            token=HF_TOKEN or None,
        )
        shutil.copy(downloaded, DATA_FILE)


DEFAULT_ANNOTATORS = [
    "Nagbhushan",
    "Divya Kempannavar",
    "Harshali",
    "Girija",
    "Asmita",
]

VERDICTS = [
    "Not judged",
    "Photo supports the review",
    "Photo contradicts the review",
    "Photo shows something the review never mentions",
    "Cannot tell from this photo",
]

IMAGE_QUALITY_LEVELS = [
    "Informative (shows product, defect, or real usage)",
    "Redundant (duplicates seller catalog photo)",
    "Noise (packaging, blurry, or unrelated)",
]

MISMATCH_PATTERN = re.compile(
    r"\b(broken|broke|crack(?:ed|s)?|scratch(?:ed|es)?|damaged|dent(?:ed)?|"
    r"wrong|different|missing|fake|counterfeit|cheap|flimsy|peeling|torn|"
    r"not as pictured|not as described|doesn['’]?t fit|didn['’]?t fit|"
    r"stopped working|fell apart|defective|poor quality|misleading|color is off)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Cross-modal review & summary checker (v2)", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1420px; }
      .review-card {
        border: 1px solid rgba(128,128,128,0.28);
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1.1rem;
      }
      .meta-line {
        font-size: 0.82rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        opacity: 0.65;
        margin-bottom: 0.2rem;
      }
      .fact-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.55;
      }
      .fact-value { font-size: 1rem; margin-bottom: 0.7rem; }
      .badge-high {
        display: inline-block;
        padding: 0.1rem 0.45rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(217, 119, 6, 0.15);
        color: #b45309;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------- helpers -----------------------------------
def normalize_annotator(name: str) -> str:
    cleaned = " ".join((name or "").strip().split())
    if not cleaned:
        return ""
    alias_map = {
        "nagbhushan": "Nagbhushan",
        "divya kempannavar": "Divya Kempannavar",
        "harshali": "Harshali",
        "girija": "Girija",
        "asmita": "Asmita",
        "asmta": "Asmita",
    }
    return alias_map.get(cleaned.lower(), cleaned.title())


def review_key(r: dict) -> str:
    return f"{r.get('user_id', '')}:{r.get('timestamp', '')}"


def contradiction_priority(r: dict) -> float:
    """Heuristic score to surface reviews likely to yield contradictions or visual-only evidence."""
    if "contradiction_score" in r:
        return float(r["contradiction_score"])
    score = 0.0
    rating = float(r.get("rating") or 5.0)
    if rating <= 2.0:
        score += 3.0
    elif rating <= 3.0:
        score += 2.0
    text = f"{r.get('title') or ''} {r.get('text') or ''}"
    hits = len(MISMATCH_PATTERN.findall(text))
    score += min(hits, 4) * 1.25
    # High rating with defect/mismatch keywords is a classic cross-modal conflict signal
    if rating >= 4.0 and hits >= 1:
        score += 2.0
    n_imgs = len(r.get("image_urls") or [])
    if n_imgs >= 2:
        score += 0.5
    return score


def show_image(url, caption=None):
    try:
        st.image(url, caption=caption, use_container_width=True)
    except TypeError:
        st.image(url, caption=caption, use_column_width=True)


# ------------------------------- data --------------------------------------
@st.cache_resource(show_spinner="Loading products...")
def load_products(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return []
    items = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


@st.cache_resource(show_spinner=False)
def build_index(path_str: str):
    return {p["parent_asin"]: p for p in load_products(path_str)}


def search(products, query, limit=40):
    words = [w for w in query.lower().split() if len(w) > 1]
    if not words:
        return []
    scored = []
    for p in products:
        title = (p.get("title") or "").lower()
        hits = sum(1 for w in words if w in title)
        if hits:
            phrase = 1 if query.lower() in title else 0
            scored.append((phrase, hits / len(words), -len(title), p))
    scored.sort(key=lambda t: t[:3], reverse=True)
    return [t[3] for t in scored[:limit]]


# ------------------------------- database ----------------------------------
@st.cache_resource(show_spinner=False)
def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS annotations (
            annotator     TEXT NOT NULL,
            parent_asin   TEXT NOT NULL,
            review_key    TEXT NOT NULL,
            product_title TEXT,
            review_text   TEXT,
            image_urls    TEXT,
            verdict       TEXT,
            image_quality TEXT,
            claim         TEXT,
            notes         TEXT,
            updated_at    TEXT,
            PRIMARY KEY (annotator, parent_asin, review_key)
        )
        """
    )
    # Backward-compatible migration for existing v1 databases
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(annotations)").fetchall()
    }
    if "image_quality" not in existing_cols:
        conn.execute("ALTER TABLE annotations ADD COLUMN image_quality TEXT")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS product_summaries (
            annotator             TEXT NOT NULL,
            parent_asin           TEXT NOT NULL,
            product_title         TEXT,
            gold_summary          TEXT NOT NULL,
            pros                  TEXT,
            cons                  TEXT,
            visual_only_findings  TEXT,
            cross_modal_conflicts TEXT,
            notes                 TEXT,
            updated_at            TEXT,
            PRIMARY KEY (annotator, parent_asin)
        )
        """
    )
    conn.commit()
    return conn


def save_annotation(conn, **row):
    row["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    conn.execute(
        f"INSERT OR REPLACE INTO annotations ({cols}) VALUES ({marks})",
        list(row.values()),
    )
    conn.commit()


def load_annotation(conn, annotator, parent_asin, rk):
    cur = conn.execute(
        "SELECT verdict, image_quality, claim, notes FROM annotations "
        "WHERE annotator=? AND parent_asin=? AND review_key=?",
        (annotator, parent_asin, rk),
    )
    return cur.fetchone()


def save_product_summary(conn, **row):
    row["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    conn.execute(
        f"INSERT OR REPLACE INTO product_summaries ({cols}) VALUES ({marks})",
        list(row.values()),
    )
    conn.commit()


def load_product_summary(conn, annotator, parent_asin):
    cur = conn.execute(
        "SELECT gold_summary, pros, cons, visual_only_findings, "
        "cross_modal_conflicts, notes FROM product_summaries "
        "WHERE annotator=? AND parent_asin=?",
        (annotator, parent_asin),
    )
    return cur.fetchone()


def export_table_csv(conn, table_name: str, order_by: str) -> str:
    cur = conn.execute(f"SELECT * FROM {table_name} ORDER BY {order_by}")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([d[0] for d in cur.description])
    writer.writerows(cur.fetchall())
    return buf.getvalue()


def get_iaa_candidates(conn, current_annotator: str, limit: int = 50):
    """Return (parent_asin, review_key) pairs judged by exactly 1 other annotator."""
    if not current_annotator:
        return []
    rows = conn.execute(
        """
        SELECT parent_asin, review_key
        FROM annotations
        WHERE verdict IS NOT NULL AND verdict != 'Not judged'
        GROUP BY parent_asin, review_key
        HAVING COUNT(DISTINCT annotator) = 1
           AND MAX(annotator) != ?
        LIMIT ?
        """,
        (current_annotator, limit),
    ).fetchall()
    return rows


def compute_cohens_kappa(pairs):
    """Compute unweighted Cohen's kappa for a list of (label_a, label_b) tuples."""
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


# --------------------------------- app -------------------------------------
conn = get_db()
ensure_products_file()
products = load_products(str(DATA_FILE))
index = build_index(str(DATA_FILE))

# Build canonical annotator list from defaults + DB
db_annotators = [
    normalize_annotator(r[0])
    for r in conn.execute("SELECT DISTINCT annotator FROM annotations").fetchall()
    if r[0] and r[0].lower() != "annotator"
]
known_annotators = sorted(set(DEFAULT_ANNOTATORS) | set(db_annotators))

with st.sidebar:
    st.subheader("Annotator")
    choice = st.selectbox(
        "Select your name",
        options=["-- Select --"] + known_annotators + ["+ Add new annotator"],
        index=0,
    )
    if choice == "+ Add new annotator":
        raw_new = st.text_input("Enter full name", placeholder="e.g. Rahul Sharma")
        annotator = normalize_annotator(raw_new)
    elif choice == "-- Select --":
        annotator = ""
    else:
        annotator = choice

    st.divider()
    st.caption(f"{len(products):,} products loaded")
    done = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE verdict != 'Not judged'"
    ).fetchone()[0]
    mine = (
        conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE annotator=? AND verdict != 'Not judged'",
            (annotator,),
        ).fetchone()[0]
        if annotator
        else 0
    )
    sum_done = conn.execute("SELECT COUNT(*) FROM product_summaries").fetchone()[0]
    st.metric("Review judgements", f"{done:,}", delta=f"{mine:,} by you")
    st.metric("Product summaries", f"{sum_done:,}")

    st.download_button(
        "Download review judgements (CSV)",
        data=export_table_csv(conn, "annotations", "annotator, parent_asin, review_key"),
        file_name="cross_modal_judgements_v2.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Download product summaries (CSV)",
        data=export_table_csv(conn, "product_summaries", "parent_asin, annotator"),
        file_name="product_summaries_v2.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.title("Cross-modal review & summary checker (v2)")

if not products:
    st.error(
        f"No product file at {DATA_FILE}. Run prepare_app_data.py first, then "
        "place products.jsonl beside this script or configure HF_DATASET_REPO."
    )
    st.stop()

tab_reviews, tab_summaries, tab_audit = st.tabs(
    ["1. Review Faithfulness", "2. Product Summaries", "3. Dataset Audit & IAA"]
)

# ======================== TAB 1: REVIEW FAITHFULNESS =======================
with tab_reviews:
    mode = st.radio(
        "Sampling mode",
        [
            "Contradiction-Rich Queue (1-3★ & mismatch keywords)",
            "Blind IAA Queue (double-annotate another annotator's review)",
            "Search / Random Product",
        ],
        horizontal=True,
    )

    selected = st.session_state.get("selected")
    iaa_target_rk = None

    if mode.startswith("Contradiction-Rich"):
        st.caption(
            "Prioritizes products containing low-rating or defect/mismatch-keyword reviews "
            "to build balanced Contradiction and Visual-Only classes."
        )
        c1, c2 = st.columns([4, 1])
        if c2.button("Next high-signal product", use_container_width=True):
            high_signal = [
                p
                for p in products
                if any(contradiction_priority(r) >= 3.0 for r in p.get("reviews", []))
            ]
            pool = high_signal or products
            st.session_state["selected"] = random.choice(pool)["parent_asin"]
            selected = st.session_state["selected"]
        if not selected:
            high_signal = [
                p
                for p in products
                if any(contradiction_priority(r) >= 3.0 for r in p.get("reviews", []))
            ]
            if high_signal:
                selected = high_signal[0]["parent_asin"]
                st.session_state["selected"] = selected

    elif mode.startswith("Blind IAA"):
        if not annotator:
            st.info("Select your name in the sidebar to load your blind IAA queue.")
            st.stop()
        candidates = get_iaa_candidates(conn, annotator, limit=100)
        candidates = [(pa, rk) for pa, rk in candidates if pa in index]
        if not candidates:
            st.success(
                "No pending blind IAA reviews right now (all single-annotated reviews "
                "were either done by you or already double-annotated)."
            )
            st.stop()
        st.info(
            f"{len(candidates)} reviews available for blind double-annotation. "
            "Prior annotator labels are hidden."
        )
        picked_idx = st.selectbox(
            "Pick an IAA review",
            options=list(range(len(candidates))),
            format_func=lambda idx: f"{index[candidates[idx][0]]['title'][:85]} ({candidates[idx][1][:14]}...)",
        )
        selected, iaa_target_rk = candidates[picked_idx]
        st.session_state["selected"] = selected

    else:
        with st.form("search_form"):
            c1, c2, c3 = st.columns([6, 1, 1])
            query = c1.text_input(
                "Product name",
                value=st.session_state.get("query", ""),
                placeholder="Paste or type a product name",
                label_visibility="collapsed",
            )
            submitted = c2.form_submit_button("Submit", use_container_width=True)
            surprise = c3.form_submit_button("Random", use_container_width=True)

        if surprise:
            st.session_state["selected"] = random.choice(products)["parent_asin"]
            st.session_state["matches"] = None
        elif submitted:
            st.session_state["query"] = query
            matches = search(products, query)
            st.session_state["matches"] = [p["parent_asin"] for p in matches]
            st.session_state["selected"] = matches[0]["parent_asin"] if matches else None
            if not matches:
                st.warning("No product title contains those words. Try broader words.")

        match_ids = st.session_state.get("matches")
        if match_ids and len(match_ids) > 1:
            chosen = st.selectbox(
                f"{len(match_ids)} products matched - pick one",
                options=match_ids,
                format_func=lambda pa: index[pa]["title"][:110],
                index=match_ids.index(st.session_state["selected"])
                if st.session_state.get("selected") in match_ids
                else 0,
            )
            st.session_state["selected"] = chosen
        selected = st.session_state.get("selected")

    if selected and selected in index:
        p = index[selected]
        st.divider()
        left, right = st.columns([2, 3], gap="large")
        with left:
            if p.get("catalog_images"):
                show_image(p["catalog_images"][0])
                if len(p["catalog_images"]) > 1:
                    cols = st.columns(min(3, len(p["catalog_images"]) - 1))
                    for col, url in zip(cols, p["catalog_images"][1:4]):
                        with col:
                            show_image(url)
            else:
                st.info("This product has no catalog photo in the metadata.")
            st.caption("Seller photos. Reviewer photos appear beside each review.")

        with right:
            st.subheader(p["title"])
            f1, f2, f3 = st.columns(3)
            f1.markdown(
                f"<div class='fact-label'>Rating</div><div class='fact-value'>"
                f"{p.get('average_rating', '-')} ({p.get('rating_number') or 0:,})</div>",
                unsafe_allow_html=True,
            )
            f2.markdown(
                f"<div class='fact-label'>Price</div><div class='fact-value'>"
                f"{p.get('price') or '-'}</div>",
                unsafe_allow_html=True,
            )
            f3.markdown(
                f"<div class='fact-label'>Store</div><div class='fact-value'>"
                f"{p.get('store') or '-'}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"parent_asin `{p['parent_asin']}`")
            if p.get("features"):
                with st.expander("Listed features"):
                    for feat in p["features"]:
                        st.markdown(f"- {feat}")
            if p.get("description"):
                with st.expander("Description"):
                    st.write(p["description"])

        st.divider()
        revs = list(p.get("reviews", []))
        if iaa_target_rk:
            revs = [r for r in revs if review_key(r) == iaa_target_rk]
        elif mode.startswith("Contradiction-Rich"):
            revs.sort(key=contradiction_priority, reverse=True)

        st.subheader(f"Reviews with photos ({len(revs)})")
        if not annotator:
            st.info("Select your name in the sidebar to record judgements.")

        for i, r in enumerate(revs):
            rk = review_key(r)
            prio = contradiction_priority(r)
            st.markdown("<div class='review-card'>", unsafe_allow_html=True)
            rc_left, rc_right = st.columns([2, 3], gap="large")

            with rc_left:
                urls = r.get("image_urls", [])
                for url in urls[:3]:
                    show_image(url)
                if len(urls) > 3:
                    st.caption(f"+{len(urls) - 3} more photos on this review")

            with rc_right:
                stars = "★" * int(r.get("rating") or 0) + "☆" * (5 - int(r.get("rating") or 0))
                badge = (
                    " &nbsp;·&nbsp; <span class='badge-high'>High mismatch signal</span>"
                    if prio >= 3.0
                    else ""
                )
                st.markdown(
                    f"<div class='meta-line'>{stars} &nbsp;·&nbsp; "
                    f"{'verified purchase' if r.get('verified_purchase') else 'unverified'} "
                    f"&nbsp;·&nbsp; {r.get('helpful_vote') or 0} found helpful{badge}</div>",
                    unsafe_allow_html=True,
                )
                if r.get("title"):
                    st.markdown(f"**{r['title']}**")
                st.write(r.get("text", ""))

                if annotator:
                    saved = load_annotation(conn, annotator, p["parent_asin"], rk)
                    prev_v, prev_iq, prev_c, prev_n = (
                        saved if saved else ("Not judged", IMAGE_QUALITY_LEVELS[0], "", "")
                    )
                    verdict = st.radio(
                        "Do the photo and the words agree?",
                        VERDICTS,
                        index=VERDICTS.index(prev_v) if prev_v in VERDICTS else 0,
                        key=f"v_{p['parent_asin']}_{rk}_{i}",
                    )
                    image_quality = st.selectbox(
                        "Image informativeness (for visual gating)",
                        IMAGE_QUALITY_LEVELS,
                        index=IMAGE_QUALITY_LEVELS.index(prev_iq)
                        if prev_iq in IMAGE_QUALITY_LEVELS
                        else 0,
                        key=f"iq_{p['parent_asin']}_{rk}_{i}",
                    )
                    claim = st.text_input(
                        "Exact sentence or claim the photo bears on",
                        value=prev_c or "",
                        key=f"c_{p['parent_asin']}_{rk}_{i}",
                        placeholder="Copy the exact phrase, e.g. 'broke in two pieces after 3 days'",
                    )
                    notes = st.text_input(
                        "Evidence / discrepancy notes (required for Contradicts & Unmentioned)",
                        value=prev_n or "",
                        key=f"n_{p['parent_asin']}_{rk}_{i}",
                    )
                    if st.button("Save judgement", key=f"s_{p['parent_asin']}_{rk}_{i}"):
                        if verdict == "Not judged":
                            st.error("Select a verdict before saving.")
                        elif verdict in (
                            "Photo supports the review",
                            "Photo contradicts the review",
                            "Photo shows something the review never mentions",
                        ) and not claim.strip():
                            st.error("Please fill in the claim phrase before saving.")
                        elif verdict in (
                            "Photo contradicts the review",
                            "Photo shows something the review never mentions",
                        ) and not notes.strip():
                            st.error(
                                "Please add a short note describing what the photo shows."
                            )
                        else:
                            save_annotation(
                                conn,
                                annotator=annotator,
                                parent_asin=p["parent_asin"],
                                review_key=rk,
                                product_title=p["title"],
                                review_text=r.get("text", ""),
                                image_urls=json.dumps(r.get("image_urls", [])),
                                verdict=verdict,
                                image_quality=image_quality,
                                claim=claim.strip(),
                                notes=notes.strip(),
                            )
                            st.success("Saved.")

            st.markdown("</div>", unsafe_allow_html=True)

# ======================== TAB 2: PRODUCT SUMMARIES =========================
with tab_summaries:
    st.subheader("Product-Level Multimodal Gold Summary Authoring")
    st.write(
        "Use judged products (with 5+ image reviews) to write concise, faithful multimodal "
        "summaries grounded in both customer text and buyer photos."
    )
    if not annotator:
        st.info("Select your name in the sidebar to author product summaries.")
    else:
        annotated_asins = [
            row[0]
            for row in conn.execute(
                """
                SELECT parent_asin
                FROM annotations
                WHERE verdict != 'Not judged'
                GROUP BY parent_asin
                HAVING COUNT(DISTINCT review_key) >= 5
                ORDER BY COUNT(DISTINCT review_key) DESC
                """
            ).fetchall()
            if row[0] in index
        ]
        summary_pool = annotated_asins or [p["parent_asin"] for p in products[:50]]
        chosen_asin = st.selectbox(
            f"Choose product ({len(annotated_asins)} with >=5 judged reviews)",
            options=summary_pool,
            format_func=lambda pa: f"{pa} — {index[pa]['title'][:95]}",
        )
        prod = index[chosen_asin]
        prev_sum = load_product_summary(conn, annotator, chosen_asin)
        p_gold, p_pros, p_cons, p_vis, p_conf, p_notes = (
            prev_sum if prev_sum else ("", "", "", "", "", "")
        )

        col_ev, col_form = st.columns([1, 1], gap="large")
        with col_ev:
            st.markdown(f"**{prod['title']}**")
            st.caption(f"{len(prod.get('reviews', []))} reviews with photos")
            for r in prod.get("reviews", [])[:12]:
                stars = "★" * int(r.get("rating") or 0)
                with st.expander(f"{stars} — {(r.get('title') or r.get('text', ''))[:70]}"):
                    st.write(r.get("text", ""))
                    if r.get("image_urls"):
                        show_image(r["image_urls"][0])

        with col_form:
            gold_summary = st.text_area(
                "Gold Multimodal Summary (3–5 sentences, strictly faithful to text + photos)",
                value=p_gold,
                height=140,
            )
            pros = st.text_area("Key Pros (supported by reviews/photos)", value=p_pros, height=70)
            cons = st.text_area("Key Cons (supported by reviews/photos)", value=p_cons, height=70)
            visual_only = st.text_area(
                "Visual-Only Findings (visible in photos, omitted in text)",
                value=p_vis,
                height=70,
            )
            conflicts = st.text_area(
                "Cross-Modal Conflicts (where text and buyer photos disagree)",
                value=p_conf,
                height=70,
            )
            s_notes = st.text_input("Notes", value=p_notes)
            if st.button("Save product summary", type="primary"):
                if len(gold_summary.strip()) < 40:
                    st.error("Please write a substantive gold summary (at least 40 characters).")
                else:
                    save_product_summary(
                        conn,
                        annotator=annotator,
                        parent_asin=chosen_asin,
                        product_title=prod["title"],
                        gold_summary=gold_summary.strip(),
                        pros=pros.strip(),
                        cons=cons.strip(),
                        visual_only_findings=visual_only.strip(),
                        cross_modal_conflicts=conflicts.strip(),
                        notes=s_notes.strip(),
                    )
                    st.success("Product summary saved.")

# ======================== TAB 3: DATASET AUDIT & IAA =======================
with tab_audit:
    st.subheader("Live Dataset Audit & PhD Target Progress")
    rows = conn.execute(
        "SELECT annotator, parent_asin, review_key, verdict FROM annotations "
        "WHERE verdict != 'Not judged'"
    ).fetchall()

    unique_reviews = len({r[2] for r in rows})
    unique_products = len({r[1] for r in rows})
    verdict_counts = Counter(r[3] for r in rows)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unique reviews", f"{unique_reviews:,}", delta="Target: 2,500–3,000")
    m2.metric("Unique products", f"{unique_products:,}", delta="Target: 150–200")
    m3.metric(
        "Contradictions",
        f"{verdict_counts['Photo contradicts the review']:,}",
        delta="Target: 250–300",
    )
    m4.metric(
        "Visual-Only (Unmentioned)",
        f"{verdict_counts['Photo shows something the review never mentions']:,}",
        delta="Target: 250–300",
    )

    st.markdown("**Verdict distribution**")
    if rows:
        dist_rows = [
            {"Verdict": v, "Count": c, "Share (%)": round(c / len(rows) * 100, 1)}
            for v, c in verdict_counts.most_common()
        ]
        st.dataframe(dist_rows, use_container_width=True, hide_index=True)

    # Inter-annotator agreement across multi-annotated reviews
    by_rk = {}
    for ann, _, rk, v in rows:
        by_rk.setdefault(rk, {})[ann] = v

    pairs = []
    for rk, amap in by_rk.items():
        if len(amap) >= 2:
            vals = list(amap.values())
            pairs.append((vals[0], vals[1]))

    kappa, exact_acc = compute_cohens_kappa(pairs)
    st.markdown("**Inter-Annotator Agreement (Double-Annotated Subset)**")
    k1, k2, k3 = st.columns(3)
    k1.metric("Double-annotated reviews", f"{len(pairs):,}", delta="Target: 250–300")
    k2.metric("Exact agreement", f"{exact_acc * 100:.1f}%" if pairs else "—")
    k3.metric("Cohen's kappa", f"{kappa:.3f}" if kappa is not None else "—", delta="Target: >= 0.70")
