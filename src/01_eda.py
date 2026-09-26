"""
Amazon ML Challenge 2026
Business Entity Resolution
EDA - Exploratory Data Analysis

This script:
1. Inspects all training and test TSV files
2. Reports row counts and columns
3. Checks missing/empty values
4. Checks duplicate entity IDs
5. Shows country distributions
6. Analyzes name/address lengths
7. Analyzes ground-truth match counts
8. Reports Source 2 / Source 3 match distribution
9. Saves a compact EDA report

IMPORTANT:
- Challenge files are TSV, so sep="\\t" is used.
- The original dataset is never modified.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# Project root = folder containing this script's parent directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"

OUTPUT_DIR = PROJECT_ROOT / "output"
EDA_DIR = OUTPUT_DIR / "eda"

EDA_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILES = {
    "source1": TRAIN_DIR / "train_source1.tsv",
    "source2": TRAIN_DIR / "train_source2.tsv",
    "source3": TRAIN_DIR / "train_source3.tsv",
    "ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
}

TEST_FILES = {
    "source1": TEST_DIR / "test_source1.tsv",
    "source2": TEST_DIR / "test_source2.tsv",
    "source3": TEST_DIR / "test_source3.tsv",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    """Print a clear section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_file(path):
    """Check whether a file exists."""
    if not path.exists():
        print(f"WARNING: File not found: {path}")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)

    print(f"File: {path}")
    print(f"Size: {size_mb:.2f} MB")

    return True


def load_tsv(path):
    """
    Load a TSV file.

    The challenge explicitly requires tab-separated reading.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


def basic_dataset_info(df, name):
    """Print basic information about a dataset."""

    print_header(f"BASIC INFORMATION: {name}")

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


def missing_value_analysis(df, name):
    """Analyze missing and empty values."""

    print_header(f"MISSING / EMPTY VALUES: {name}")

    rows = []

    for col in df.columns:
        empty_count = (df[col].astype(str).str.strip() == "").sum()

        rows.append({
            "column": col,
            "empty_count": empty_count,
            "empty_percentage": (
                empty_count / len(df) * 100
                if len(df) > 0
                else 0
            )
        })

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    return result


def duplicate_id_analysis(df, name):
    """Check duplicate entity IDs."""

    print_header(f"DUPLICATE ID ANALYSIS: {name}")

    if "entity_id" not in df.columns:
        print("No entity_id column found.")
        return

    duplicate_count = df["entity_id"].duplicated().sum()

    print(f"Duplicate entity_id rows: {duplicate_count:,}")

    if duplicate_count > 0:
        duplicates = df[
            df["entity_id"].duplicated(keep=False)
        ].sort_values("entity_id")

        print("\nExample duplicates:")
        print(duplicates.head(20).to_string(index=False))


def country_analysis(df, name):
    """Analyze country distribution."""

    print_header(f"COUNTRY DISTRIBUTION: {name}")

    if "country" not in df.columns:
        print("No country column found.")
        return

    counts = df["country"].value_counts(dropna=False)

    percentages = (
        df["country"]
        .value_counts(normalize=True, dropna=False)
        * 100
    )

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2)
    })

    print(result.to_string())


def text_length_analysis(df, name):
    """Analyze business name and address lengths."""

    print_header(f"TEXT LENGTH ANALYSIS: {name}")

    columns = [
        "business_name",
        "business_address"
    ]

    results = []

    for col in columns:

        if col not in df.columns:
            continue

        lengths = df[col].astype(str).str.len()

        results.append({
            "field": col,
            "min": lengths.min(),
            "mean": round(lengths.mean(), 2),
            "median": lengths.median(),
            "max": lengths.max(),
            "p25": lengths.quantile(0.25),
            "p75": lengths.quantile(0.75),
        })

    result = pd.DataFrame(results)

    print(result.to_string(index=False))


def unusual_text_examples(df, name):
    """
    Display examples that may be useful for understanding
    real-world noise.
    """

    print_header(f"TEXT EXAMPLES: {name}")

    columns = [
        "business_name",
        "business_address"
    ]

    for col in columns:

        if col not in df.columns:
            continue

        print(f"\n--- {col} examples ---")

        sample = (
            df[col]
            .drop_duplicates()
            .sample(
                min(20, df[col].nunique()),
                random_state=42
            )
        )

        for value in sample:
            print(f"  {value}")


# ============================================================
# GROUND TRUTH ANALYSIS
# ============================================================

def ground_truth_analysis(gt):
    """
    Analyze train_ground_truth.tsv.

    matched_entity_ids is a comma-separated list.
    Empty means the Source 1 entity has no matches.
    """

    print_header("GROUND TRUTH ANALYSIS")

    print(f"Ground truth rows: {len(gt):,}")

    if "source1_entity_id" not in gt.columns:
        print("ERROR: source1_entity_id column not found.")
        return

    if "matched_entity_ids" not in gt.columns:
        print("ERROR: matched_entity_ids column not found.")
        return

    # Normalize empty values
    matches = gt["matched_entity_ids"].fillna("").astype(str).str.strip()

    # Number of matched IDs per Source 1 entity
    match_counts = matches.apply(
        lambda x: 0 if x == "" else len(x.split(","))
    )

    print("\nMatch count statistics:")
    print(match_counts.describe().to_string())

    print("\nDistribution of number of matches per Source 1 entity:")

    distribution = (
        match_counts
        .value_counts()
        .sort_index()
        .rename_axis("number_of_matches")
        .reset_index(name="source1_entities")
    )

    distribution["percentage"] = (
        distribution["source1_entities"]
        / len(gt)
        * 100
    ).round(2)

    print(distribution.to_string(index=False))

    # Singleton count
    singleton_count = (match_counts == 0).sum()

    print("\nEntities with zero matches:")
    print(f"  {singleton_count:,}")
    print(
        f"  {singleton_count / len(gt) * 100:.2f}%"
        if len(gt) > 0
        else "  N/A"
    )

    # --------------------------------------------------------
    # Count S2 vs S3 matches
    # --------------------------------------------------------

    s2_count = 0
    s3_count = 0
    invalid_count = 0

    for value in matches:

        if value == "":
            continue

        ids = [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

        for entity_id in ids:

            if entity_id.startswith("S2-"):
                s2_count += 1

            elif entity_id.startswith("S3-"):
                s3_count += 1

            else:
                invalid_count += 1

    print("\nGround truth matched IDs:")
    print(f"  S2 matches : {s2_count:,}")
    print(f"  S3 matches : {s3_count:,}")
    print(f"  Other IDs  : {invalid_count:,}")

    # --------------------------------------------------------
    # Save distribution
    # --------------------------------------------------------

    distribution.to_csv(
        EDA_DIR / "ground_truth_match_distribution.csv",
        index=False
    )

    return distribution


# ============================================================
# CROSS-SOURCE ANALYSIS
# ============================================================

def compare_sources(source1, source2, source3):
    """
    Basic cross-source statistics.

    This does NOT perform entity matching.
    It only checks country distributions and field availability.
    """

    print_header("CROSS-SOURCE SUMMARY")

    summary = []

    for source_name, df in [
        ("Source 1", source1),
        ("Source 2", source2),
        ("Source 3", source3),
    ]:

        summary.append({
            "source": source_name,
            "rows": len(df),
            "unique_entity_ids": (
                df["entity_id"].nunique()
                if "entity_id" in df.columns
                else None
            ),
            "unique_countries": (
                df["country"].nunique()
                if "country" in df.columns
                else None
            ),
        })

    result = pd.DataFrame(summary)

    print(result.to_string(index=False))

    result.to_csv(
        EDA_DIR / "source_summary.csv",
        index=False
    )


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def analyze_source(path, name):
    """Load and analyze one source file."""

    if not check_file(path):
        return None

    df = load_tsv(path)

    basic_dataset_info(df, name)

    missing_value_analysis(df, name)

    duplicate_id_analysis(df, name)

    country_analysis(df, name)

    text_length_analysis(df, name)

    unusual_text_examples(df, name)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("BUSINESS ENTITY RESOLUTION")
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Train folder : {TRAIN_DIR}")
    print(f"Test folder  : {TEST_DIR}")
    print(f"EDA output   : {EDA_DIR}")

    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    print_header("LOADING TRAINING DATA")

    train_s1 = analyze_source(
        TRAIN_FILES["source1"],
        "TRAIN SOURCE 1"
    )

    train_s2 = analyze_source(
        TRAIN_FILES["source2"],
        "TRAIN SOURCE 2"
    )

    train_s3 = analyze_source(
        TRAIN_FILES["source3"],
        "TRAIN SOURCE 3"
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    print_header("LOADING GROUND TRUTH")

    if check_file(TRAIN_FILES["ground_truth"]):

        train_gt = load_tsv(
            TRAIN_FILES["ground_truth"]
        )

        basic_dataset_info(
            train_gt,
            "TRAIN GROUND TRUTH"
        )

        ground_truth_analysis(train_gt)

    else:
        train_gt = None

    # --------------------------------------------------------
    # CROSS SOURCE
    # --------------------------------------------------------

    if (
        train_s1 is not None
        and train_s2 is not None
        and train_s3 is not None
    ):
        compare_sources(
            train_s1,
            train_s2,
            train_s3
        )

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    print_header("LOADING TEST DATA")

    test_s1 = analyze_source(
        TEST_FILES["source1"],
        "TEST SOURCE 1"
    )

    test_s2 = analyze_source(
        TEST_FILES["source2"],
        "TEST SOURCE 2"
    )

    test_s3 = analyze_source(
        TEST_FILES["source3"],
        "TEST SOURCE 3"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_header("FINAL DATASET SUMMARY")

    summary = []

    datasets = [
        ("train_source1", train_s1),
        ("train_source2", train_s2),
        ("train_source3", train_s3),
        ("train_ground_truth", train_gt),
        ("test_source1", test_s1),
        ("test_source2", test_s2),
        ("test_source3", test_s3),
    ]

    for name, df in datasets:

        if df is None:
            continue

        summary.append({
            "dataset": name,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(
                df.memory_usage(deep=True).sum()
                / (1024 * 1024),
                2
            )
        })

    summary_df = pd.DataFrame(summary)

    print(summary_df.to_string(index=False))

    summary_df.to_csv(
        EDA_DIR / "dataset_summary.csv",
        index=False
    )

    print("\n")
    print("=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)

    print(f"\nReports saved to:")
    print(EDA_DIR)


if __name__ == "__main__":
    main()