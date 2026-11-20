# Multimodal Product Review Summarization & Faithfulness Annotation (v2)

Upgraded Streamlit + SQLite data collection and dataset auditing suite built for PhD research on Cross-Modal Product Review Summarization and Faithfulness Evaluation using [Amazon Reviews '23](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

## Key Upgrades over v1
1. **Controlled Annotator Selector**: Eliminates casing/typo duplicates (e.g., `Divya kempannavar`, `asmta`).
2. **Contradiction-Rich Queue**: Prioritizes 1–3★ reviews and defect/mismatch keywords to rapidly boost minority classes (`Photo contradicts the review` and `Photo shows something the review never mentions`).
3. **Blind IAA Queue**: Automatically serves single-annotated reviews from other annotators with prior labels hidden to reach 250–300 double-annotated reviews.
4. **Image Informativeness Tagging**: Records `Informative` / `Redundant` / `Noise` per review to support dynamic visual gating experiments.
5. **Product Summary Tab**: New `product_summaries` SQLite table for authoring gold multimodal product summaries (Pros, Cons, Visual-Only Findings, Cross-Modal Conflicts).
6. **Live Dataset Audit & IAA Dashboard**: Displays class distribution, PhD target progress, exact agreement, and Cohen's kappa directly inside the app.
7. **Leak-Free Cleaning & Splitting (`clean_and_split.py`)**: Cleans existing v1 CSV exports and builds 70/15/15 splits grouped strictly by `parent_asin`.

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Clean Existing v1 CSV & Build Leak-Free Splits
```bash
python clean_and_split.py --input Main_Data_Cross_Modal.csv --outdir cleaned_splits
```
