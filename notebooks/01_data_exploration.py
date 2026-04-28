"""
=============================================================================
Script 01: Exploratory Data Analysis (EDA)
=============================================================================
Purpose:
    Perform initial exploration of the raw dataset to understand its
    structure, quality issues, distributions, and potential problems
    before cleaning.

Outputs:
    - Console report of data quality assessment
    - Missing value heatmap (saved to reports/)
    - Distribution plots for key columns (saved to reports/)
=============================================================================
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.cleaning_pipeline import analyze_missing
from src.viz_utils import setup_style, save_figure, COLORS, CATEGORY_PALETTE

# Apply premium styling
setup_style()


def load_raw_data():
    """Load the raw dataset."""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "raw", "ecommerce_sales_raw.csv"
    )

    if not os.path.exists(data_path):
        print("❌ Raw dataset not found!")
        print("   Run 'python src/data_generator.py' first to generate the data.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset: {data_path}")
    return df


def basic_info(df: pd.DataFrame):
    """Print basic dataset information."""
    print("\n" + "=" * 70)
    print("  📋 BASIC DATASET INFORMATION")
    print("=" * 70)
    print(f"\n  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(f"\n  {'Column':<20} {'Type':<15} {'Non-Null':>10} {'Null':>8} {'Unique':>8}")
    print("  " + "-" * 65)
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        null = df[col].isna().sum()
        unique = df[col].nunique()
        print(f"  {col:<20} {dtype:<15} {non_null:>10,} {null:>8,} {unique:>8,}")


def missing_value_analysis(df: pd.DataFrame):
    """Analyze and visualize missing values."""
    print("\n" + "=" * 70)
    print("  🔍 MISSING VALUE ANALYSIS")
    print("=" * 70)

    summary = analyze_missing(df)

    if summary.empty:
        print("\n  ✅ No missing values found!")
        return

    print(f"\n  Total missing values: {df.isnull().sum().sum():,}")
    print(f"  Columns with missing values: {len(summary)}")
    print(f"\n{summary.to_string()}")

    # ── Missing Value Heatmap ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: Missing value matrix (sample)
    sample_size = min(200, len(df))
    sample = df.sample(sample_size, random_state=42)
    missing_matrix = sample.isnull().astype(int)

    sns.heatmap(
        missing_matrix, cbar=False, ax=axes[0],
        yticklabels=False, cmap=["#1E293B", "#EF4444"],
        linewidths=0, linecolor=COLORS["grid"]
    )
    axes[0].set_title("Missing Value Pattern (Sample of 200 rows)",
                       fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

    # Right: Missing percentage bar chart
    cols_with_missing = df.isnull().sum()
    cols_with_missing = cols_with_missing[cols_with_missing > 0].sort_values(ascending=True)
    pcts = (cols_with_missing / len(df) * 100)

    bars = axes[1].barh(
        pcts.index, pcts.values,
        color=COLORS["accent"], alpha=0.85,
        edgecolor=COLORS["grid"], linewidth=0.5
    )

    # Add percentage labels
    for bar, val in zip(bars, pcts.values):
        axes[1].text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=10,
            color=COLORS["text_secondary"], fontweight="bold"
        )

    axes[1].set_title("Missing Value Percentage by Column",
                       fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Missing %")

    fig.suptitle("Missing Value Analysis", fontsize=17, fontweight="bold",
                 color=COLORS["primary"], y=1.02)
    plt.tight_layout()
    save_figure(fig, "01_missing_values.png")
    plt.close()


def distribution_analysis(df: pd.DataFrame):
    """Analyze distributions of numeric columns."""
    print("\n" + "=" * 70)
    print("  📊 DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n  Numeric columns: {numeric_cols}")

    if not numeric_cols:
        print("  ⚠️  No numeric columns found.")
        return

    # Print statistics
    print(f"\n{df[numeric_cols].describe().round(2).to_string()}")

    # ── Distribution Plots ─────────────────────────────────────────────────
    n_cols = min(len(numeric_cols), 8)
    n_rows = (n_cols + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for i, col in enumerate(numeric_cols[:8]):
        ax = axes[i]
        data = df[col].dropna()

        # Convert to numeric, skip if not possible
        data = pd.to_numeric(data, errors="coerce").dropna()

        if len(data) == 0:
            ax.text(0.5, 0.5, f"No numeric data\nin '{col}'",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color=COLORS["muted"])
            ax.set_title(col, fontsize=12, fontweight="bold")
            continue

        # Histogram with KDE
        sns.histplot(data, ax=ax, kde=True, color=CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)],
                     alpha=0.6, edgecolor=COLORS["grid"], linewidth=0.5)

        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color=COLORS["warning"], linestyle="--",
                   linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color=COLORS["success"], linestyle="-.",
                   linewidth=1.5, label=f"Median: {median_val:.1f}")

        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Column Distributions", fontsize=17,
                 fontweight="bold", color=COLORS["primary"], y=1.02)
    plt.tight_layout()
    save_figure(fig, "01_distributions.png")
    plt.close()


def duplicate_analysis(df: pd.DataFrame):
    """Analyze duplicate rows."""
    print("\n" + "=" * 70)
    print("  🔄 DUPLICATE ANALYSIS")
    print("=" * 70)

    exact_dupes = df.duplicated().sum()
    print(f"\n  Exact duplicate rows: {exact_dupes:,} ({exact_dupes/len(df)*100:.1f}%)")

    # Check for near-duplicates on key columns
    if "order_id" in df.columns:
        order_dupes = df.duplicated(subset=["order_id"]).sum()
        print(f"  Duplicate order_ids: {order_dupes:,}")

    # Show a sample of duplicates
    if exact_dupes > 0:
        print(f"\n  Sample of duplicate rows:")
        dupes = df[df.duplicated(keep=False)].sort_values(
            by=df.columns.tolist()[:3]
        ).head(10)
        print(dupes.to_string(max_cols=8))


def categorical_analysis(df: pd.DataFrame):
    """Analyze categorical columns for inconsistencies."""
    print("\n" + "=" * 70)
    print("  🏷️ CATEGORICAL ANALYSIS")
    print("=" * 70)

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:
        unique_vals = df[col].nunique()
        print(f"\n  {col} — {unique_vals} unique values:")

        # Show value counts (top 15)
        value_counts = df[col].value_counts().head(15)
        for val, count in value_counts.items():
            pct = count / len(df) * 100
            print(f"    '{val}': {count:,} ({pct:.1f}%)")

        if unique_vals > 15:
            print(f"    ... and {unique_vals - 15} more unique values")

    # ── Category Distribution Plot ─────────────────────────────────────────
    if "category" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 7))

        cat_counts = df["category"].value_counts().head(20)
        bars = ax.barh(
            cat_counts.index[::-1], cat_counts.values[::-1],
            color=[CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]
                   for i in range(len(cat_counts))],
            alpha=0.85, edgecolor=COLORS["grid"], linewidth=0.5
        )

        ax.set_title("Category Distribution (Raw — Shows Inconsistencies)",
                      fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Count")

        fig.suptitle("Notice the duplicate categories due to formatting issues",
                     fontsize=11, color=COLORS["warning"], y=0.02, style="italic")
        plt.tight_layout()
        save_figure(fig, "01_category_distribution_raw.png")
        plt.close()


def data_quality_score(df: pd.DataFrame):
    """Calculate an overall data quality score."""
    print("\n" + "=" * 70)
    print("  📈 DATA QUALITY SCORE")
    print("=" * 70)

    total_cells = df.shape[0] * df.shape[1]
    missing_score = 1 - (df.isnull().sum().sum() / total_cells)
    duplicate_score = 1 - (df.duplicated().sum() / len(df))

    # Estimate type consistency (simplified)
    type_issues = 0
    for col in df.select_dtypes(include=["object"]).columns:
        # Check if column should be numeric
        numeric_count = pd.to_numeric(df[col], errors="coerce").notna().sum()
        if numeric_count > len(df) * 0.5:
            type_issues += 1

    type_score = 1 - (type_issues / max(len(df.columns), 1))

    overall = (missing_score * 0.4 + duplicate_score * 0.3 + type_score * 0.3) * 100

    scores = {
        "Completeness (missing values)": f"{missing_score * 100:.1f}%",
        "Uniqueness (duplicates)": f"{duplicate_score * 100:.1f}%",
        "Type Consistency": f"{type_score * 100:.1f}%",
        "Overall Quality Score": f"{overall:.1f}%",
    }

    for metric, score in scores.items():
        indicator = "✅" if float(score.rstrip("%")) > 90 else "⚠️" if float(score.rstrip("%")) > 70 else "❌"
        print(f"\n  {indicator} {metric}: {score}")

    print(f"\n  💡 Recommendation: {'Data needs significant cleaning' if overall < 80 else 'Data quality is acceptable'}")


# ── Main Execution ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  🔬 EXPLORATORY DATA ANALYSIS")
    print("  E-Commerce Sales Dataset — Raw Data Assessment")
    print("=" * 70)

    df = load_raw_data()

    basic_info(df)
    missing_value_analysis(df)
    distribution_analysis(df)
    duplicate_analysis(df)
    categorical_analysis(df)
    data_quality_score(df)

    print("\n" + "=" * 70)
    print("  ✅ EDA COMPLETE — Check reports/ for generated visualizations")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
