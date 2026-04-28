"""
=============================================================================
Script 02: Data Cleaning Pipeline
=============================================================================
Purpose:
    Apply a comprehensive cleaning pipeline to the raw e-commerce dataset.
    Demonstrates handling of all common data quality issues:
      1. Missing values (imputation & removal)
      2. Duplicates (exact & near-duplicate detection)
      3. Outliers (IQR & Z-score methods)
      4. Inconsistent formatting (text, dates, categories)
      5. Invalid data (range validation, email validation)
      6. Type conversion (strings → numeric)
      7. Feature engineering (derived columns)

Outputs:
    - Cleaned dataset saved to data/cleaned/
    - Cleaning report printed to console
    - Before/after comparison charts saved to reports/
=============================================================================
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.cleaning_pipeline import (
    analyze_missing,
    handle_missing_numeric,
    handle_missing_categorical,
    detect_outliers_iqr,
    detect_outliers_zscore,
    treat_outliers,
    find_duplicates,
    remove_duplicates,
    standardize_text,
    standardize_categories,
    parse_dates,
    clean_numeric_column,
    validate_range,
    validate_email,
    add_derived_features,
    generate_cleaning_report,
)
from src.viz_utils import setup_style, save_figure, COLORS, CATEGORY_PALETTE

setup_style()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA = os.path.join(PROJECT_DIR, "data", "raw", "ecommerce_sales_raw.csv")
CLEANED_DIR = os.path.join(PROJECT_DIR, "data", "cleaned")
CLEANED_DATA = os.path.join(CLEANED_DIR, "ecommerce_sales_cleaned.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0: Load Raw Data
# ═══════════════════════════════════════════════════════════════════════════════

def step_0_load_data():
    """Load the raw dataset."""
    print("\n" + "=" * 70)
    print("  📥 STEP 0: Loading Raw Data")
    print("=" * 70)

    if not os.path.exists(RAW_DATA):
        print("  ❌ Raw dataset not found! Run data_generator.py first.")
        sys.exit(1)

    df = pd.read_csv(RAW_DATA)
    print(f"  ✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Remove Duplicates
# ═══════════════════════════════════════════════════════════════════════════════

def step_1_remove_duplicates(df):
    """Remove exact and near-duplicate rows."""
    print("\n" + "=" * 70)
    print("  🔄 STEP 1: Removing Duplicates")
    print("=" * 70)

    before_count = len(df)

    # Remove exact duplicates
    df = remove_duplicates(df, keep="first")

    # Remove near-duplicates based on order_id (keep first occurrence)
    df = remove_duplicates(df, subset=["order_id"], keep="first")

    after_count = len(df)
    print(f"\n  📊 Before: {before_count:,} → After: {after_count:,} "
          f"(removed {before_count - after_count:,})")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Clean & Standardize Text
# ═══════════════════════════════════════════════════════════════════════════════

def step_2_standardize_text(df):
    """Standardize text formatting across all text columns."""
    print("\n" + "=" * 70)
    print("  📝 STEP 2: Standardizing Text & Categories")
    print("=" * 70)

    # Standardize customer names
    df = standardize_text(df, "customer_name", case="title")
    print("  → customer_name: Title case applied")

    # Standardize city names
    df = standardize_text(df, "city", case="title")
    print("  → city: Title case applied")

    # Standardize product names
    df = standardize_text(df, "product_name", case="title")
    print("  → product_name: Title case applied")

    # Map messy categories to clean ones
    category_mapping = {
        "electronics": "Electronics",
        "clothing": "Clothing",
        "clothng": "Clothing",       # Fix typo
        "home & kitchen": "Home & Kitchen",
        "home &  kitchen": "Home & Kitchen",
        "books": "Books",
        "book": "Books",
        "sports & outdoors": "Sports & Outdoors",
        "sports": "Sports & Outdoors",
        "beauty": "Beauty",
        "toys & games": "Toys & Games",
        "toys": "Toys & Games",
        "automotive": "Automotive",
    }
    df = standardize_categories(df, "category", category_mapping)
    print(f"  → category: Mapped to {len(set(category_mapping.values()))} standard categories")

    # Standardize payment methods
    df = standardize_text(df, "payment_method", case="title")
    print("  → payment_method: Title case applied")

    # Show cleaned categories
    if "category" in df.columns:
        print(f"\n  Cleaned categories: {df['category'].dropna().unique().tolist()}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Fix Data Types
# ═══════════════════════════════════════════════════════════════════════════════

def step_3_fix_types(df):
    """Convert columns to appropriate data types."""
    print("\n" + "=" * 70)
    print("  🔧 STEP 3: Fixing Data Types")
    print("=" * 70)

    # Clean unit_price (remove ₹, $, convert strings to float)
    df = clean_numeric_column(df, "unit_price", remove_chars=["₹", "$", ",", " "])
    print("  → unit_price: Cleaned and converted to float")

    # Clean customer_age (convert strings to numeric)
    df["customer_age"] = pd.to_numeric(df["customer_age"], errors="coerce")
    print("  → customer_age: Converted to numeric")

    # Parse dates
    df = parse_dates(df, "order_date")
    valid_dates = df["order_date"].notna().sum()
    print(f"  → order_date: Parsed to datetime ({valid_dates:,} valid dates)")

    # Ensure numeric types for other columns
    for col in ["quantity", "discount_pct", "shipping_days", "rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("  → quantity, discount_pct, shipping_days, rating: Ensured numeric")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Validate Data Ranges
# ═══════════════════════════════════════════════════════════════════════════════

def step_4_validate_ranges(df):
    """Validate that values fall within acceptable ranges."""
    print("\n" + "=" * 70)
    print("  ✅ STEP 4: Validating Data Ranges")
    print("=" * 70)

    # Price must be positive
    df = validate_range(df, "unit_price", min_val=0.01, max_val=None, action="nan")

    # Quantity must be positive
    df = validate_range(df, "quantity", min_val=1, max_val=None, action="nan")

    # Age must be realistic
    df = validate_range(df, "customer_age", min_val=13, max_val=100, action="nan")

    # Rating 1-5
    df = validate_range(df, "rating", min_val=1.0, max_val=5.0, action="nan")

    # Discount 0-100%
    df = validate_range(df, "discount_pct", min_val=0, max_val=100, action="cap")

    # Shipping days must be positive and reasonable
    df = validate_range(df, "shipping_days", min_val=1, max_val=30, action="nan")

    # Validate dates (not in the future)
    if "order_date" in df.columns:
        future_mask = df["order_date"] > pd.Timestamp.now()
        future_count = future_mask.sum()
        df.loc[future_mask, "order_date"] = pd.NaT
        print(f"  → order_date: {future_count} future dates set to NaT")

    # Validate emails
    df = validate_email(df, "customer_email")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Handle Outliers
# ═══════════════════════════════════════════════════════════════════════════════

def step_5_handle_outliers(df):
    """Detect and treat outliers in numeric columns."""
    print("\n" + "=" * 70)
    print("  📏 STEP 5: Handling Outliers")
    print("=" * 70)

    # Cap outliers in unit_price using IQR
    df = treat_outliers(df, "unit_price", method="iqr", action="cap", multiplier=1.5)

    # Cap outliers in quantity
    df = treat_outliers(df, "quantity", method="iqr", action="cap", multiplier=1.5)

    # Cap outliers in shipping_days
    df = treat_outliers(df, "shipping_days", method="iqr", action="cap", multiplier=1.5)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Handle Missing Values
# ═══════════════════════════════════════════════════════════════════════════════

def step_6_handle_missing(df):
    """Handle remaining missing values with appropriate strategies."""
    print("\n" + "=" * 70)
    print("  🩹 STEP 6: Handling Missing Values")
    print("=" * 70)

    print("\n  Missing values before imputation:")
    missing = analyze_missing(df)
    if not missing.empty:
        print(missing.to_string())

    # Numeric columns: impute with median
    numeric_missing = ["unit_price", "quantity", "customer_age",
                       "shipping_days", "rating", "discount_pct"]
    for col in numeric_missing:
        if col in df.columns and df[col].isnull().any():
            df = handle_missing_numeric(df, col, strategy="median")
            print(f"  → {col}: Imputed with median")

    # Categorical columns: impute with mode or 'Unknown'
    df = handle_missing_categorical(df, "category", strategy="mode")
    print("  → category: Imputed with mode")

    df = handle_missing_categorical(df, "customer_name", strategy="unknown")
    print("  → customer_name: Filled with 'Unknown'")

    df = handle_missing_categorical(df, "city", strategy="mode")
    print("  → city: Imputed with mode")

    df = handle_missing_categorical(df, "payment_method", strategy="mode")
    print("  → payment_method: Imputed with mode")

    # Drop rows with missing dates (critical column)
    if df["order_date"].isnull().any():
        before = len(df)
        df = df.dropna(subset=["order_date"])
        print(f"  → order_date: Dropped {before - len(df)} rows with missing dates")

    # Email: keep as NaN (optional field)
    print("  → customer_email: Kept as NaN (optional field)")

    print(f"\n  Missing values after imputation: {df.isnull().sum().sum()}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

def step_7_feature_engineering(df):
    """Create derived features for analysis."""
    print("\n" + "=" * 70)
    print("  ⚙️ STEP 7: Feature Engineering")
    print("=" * 70)

    df = add_derived_features(df)

    new_cols = [c for c in df.columns if c not in [
        "order_id", "order_date", "customer_name", "customer_email",
        "customer_age", "city", "category", "product_name", "quantity",
        "unit_price", "discount_pct", "payment_method", "shipping_days", "rating"
    ]]

    print(f"  ✅ Added {len(new_cols)} derived columns: {new_cols}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: Before/After Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def create_before_after_charts(df_before, df_after):
    """Create comparison visualizations showing the impact of cleaning."""
    print("\n" + "=" * 70)
    print("  📊 Creating Before/After Comparison Charts")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # ── Chart 1: Missing Values Comparison ─────────────────────────────────
    ax = axes[0, 0]
    cols = [c for c in df_before.columns if c in df_after.columns]
    before_missing = df_before[cols].isnull().sum()
    after_missing = df_after[cols].isnull().sum()

    # Only show columns that had missing values
    has_missing = before_missing[before_missing > 0].index.tolist()
    if has_missing:
        x = range(len(has_missing))
        width = 0.35
        ax.bar([i - width/2 for i in x], before_missing[has_missing].values,
               width, label="Before", color=COLORS["danger"], alpha=0.8)
        after_vals = [after_missing[c] if c in after_missing.index else 0 for c in has_missing]
        ax.bar([i + width/2 for i in x], after_vals,
               width, label="After", color=COLORS["success"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(has_missing, rotation=45, ha="right")
        ax.set_title("Missing Values: Before vs After", fontweight="bold")
        ax.legend()

    # ── Chart 2: Record Count ──────────────────────────────────────────────
    ax = axes[0, 1]
    counts = [len(df_before), len(df_after)]
    bars = ax.bar(["Raw Data", "Cleaned Data"], counts,
                  color=[COLORS["danger"], COLORS["success"]], alpha=0.8,
                  edgecolor=COLORS["grid"], linewidth=0.5)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,}", ha="center", fontsize=13, fontweight="bold",
                color=COLORS["text"])
    ax.set_title("Record Count: Before vs After", fontweight="bold")

    # ── Chart 3: Unit Price Distribution Before/After ──────────────────────
    ax = axes[1, 0]
    before_price = pd.to_numeric(df_before["unit_price"], errors="coerce").dropna()
    after_price = df_after["unit_price"].dropna()

    if len(before_price) > 0:
        ax.hist(before_price, bins=50, alpha=0.5, color=COLORS["danger"],
                label="Before", density=True)
    if len(after_price) > 0:
        ax.hist(after_price, bins=50, alpha=0.5, color=COLORS["success"],
                label="After", density=True)
    ax.set_title("Unit Price Distribution: Before vs After", fontweight="bold")
    ax.set_xlabel("Unit Price (₹)")
    ax.legend()

    # ── Chart 4: Data Quality Score ────────────────────────────────────────
    ax = axes[1, 1]

    def calc_quality(d):
        total = d.shape[0] * d.shape[1]
        completeness = 1 - (d.isnull().sum().sum() / total) if total > 0 else 0
        uniqueness = 1 - (d.duplicated().sum() / len(d)) if len(d) > 0 else 0
        return completeness * 100, uniqueness * 100

    before_comp, before_uniq = calc_quality(df_before)
    after_comp, after_uniq = calc_quality(df_after)

    categories = ["Completeness", "Uniqueness"]
    before_scores = [before_comp, before_uniq]
    after_scores = [after_comp, after_uniq]

    x = range(len(categories))
    width = 0.35
    ax.bar([i - width/2 for i in x], before_scores, width,
           label="Before", color=COLORS["danger"], alpha=0.8)
    ax.bar([i + width/2 for i in x], after_scores, width,
           label="After", color=COLORS["success"], alpha=0.8)

    for i, (b, a) in enumerate(zip(before_scores, after_scores)):
        ax.text(i - width/2, b + 1, f"{b:.1f}%", ha="center", fontsize=9,
                color=COLORS["text_secondary"])
        ax.text(i + width/2, a + 1, f"{a:.1f}%", ha="center", fontsize=9,
                color=COLORS["text_secondary"])

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 110)
    ax.set_title("Data Quality Scores: Before vs After", fontweight="bold")
    ax.legend()

    fig.suptitle("Data Cleaning Impact Analysis", fontsize=18,
                 fontweight="bold", color=COLORS["primary"], y=1.02)
    plt.tight_layout()
    save_figure(fig, "02_cleaning_comparison.png")
    plt.close()

    print("  ✅ Before/after comparison charts saved")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  🧹 DATA CLEANING PIPELINE")
    print("  E-Commerce Sales Dataset")
    print("=" * 70)

    # Load raw data
    df_raw = step_0_load_data()
    df = df_raw.copy()

    # Execute cleaning steps
    df = step_1_remove_duplicates(df)
    df = step_2_standardize_text(df)
    df = step_3_fix_types(df)
    df = step_4_validate_ranges(df)
    df = step_5_handle_outliers(df)
    df = step_6_handle_missing(df)
    df = step_7_feature_engineering(df)

    # ── Save cleaned data ──────────────────────────────────────────────────
    os.makedirs(CLEANED_DIR, exist_ok=True)
    df.to_csv(CLEANED_DATA, index=False)
    print(f"\n  💾 Cleaned dataset saved to: {CLEANED_DATA}")
    print(f"  📊 Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Generate report ────────────────────────────────────────────────────
    report = generate_cleaning_report(df_raw, df)
    print(f"\n{report}")

    # ── Create comparison charts ───────────────────────────────────────────
    create_before_after_charts(df_raw, df)

    # ── Final data preview ─────────────────────────────────────────────────
    print("\n  📋 Cleaned Data Preview (first 5 rows):")
    print(df.head().to_string())

    print("\n  📋 Cleaned Data Types:")
    print(df.dtypes.to_string())

    print("\n" + "=" * 70)
    print("  ✅ CLEANING PIPELINE COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
