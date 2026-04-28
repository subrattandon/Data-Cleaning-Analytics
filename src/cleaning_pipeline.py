"""
=============================================================================
Data Cleaning Pipeline — Reusable Functions
=============================================================================
A collection of modular, reusable functions for common data cleaning tasks.
Each function is documented with its purpose, parameters, and approach.

This module serves as a reference library for data cleaning best practices.
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Union
import re
import warnings

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MISSING VALUE HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in a DataFrame.

    Returns a summary DataFrame with count, percentage, and data type
    for each column that has missing values.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    dtypes = df.dtypes

    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct,
        "dtype": dtypes
    })

    # Only show columns with missing values, sorted by percentage
    summary = summary[summary["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )

    return summary


def handle_missing_numeric(
    df: pd.DataFrame,
    column: str,
    strategy: str = "median",
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Handle missing values in numeric columns.

    Parameters:
        df: Input DataFrame
        column: Column name to process
        strategy: One of 'mean', 'median', 'mode', 'drop', 'fill', 'interpolate'
        fill_value: Value to use when strategy='fill'

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    if strategy == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif strategy == "median":
        df[column] = df[column].fillna(df[column].median())
    elif strategy == "mode":
        df[column] = df[column].fillna(df[column].mode()[0])
    elif strategy == "drop":
        df = df.dropna(subset=[column])
    elif strategy == "fill":
        df[column] = df[column].fillna(fill_value)
    elif strategy == "interpolate":
        df[column] = df[column].interpolate(method="linear")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df


def handle_missing_categorical(
    df: pd.DataFrame,
    column: str,
    strategy: str = "mode",
    fill_value: Optional[str] = None
) -> pd.DataFrame:
    """
    Handle missing values in categorical columns.

    Parameters:
        df: Input DataFrame
        column: Column name to process
        strategy: One of 'mode', 'drop', 'fill', 'unknown'
        fill_value: Value to use when strategy='fill'
    """
    df = df.copy()

    if strategy == "mode":
        df[column] = df[column].fillna(df[column].mode()[0])
    elif strategy == "drop":
        df = df.dropna(subset=[column])
    elif strategy == "fill":
        df[column] = df[column].fillna(fill_value)
    elif strategy == "unknown":
        df[column] = df[column].fillna("Unknown")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OUTLIER DETECTION & TREATMENT
# ═══════════════════════════════════════════════════════════════════════════════

def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> pd.Series:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Returns a boolean Series where True indicates an outlier.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return (df[column] < lower_bound) | (df[column] > upper_bound)


def detect_outliers_zscore(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers using the Z-score method.

    Returns a boolean Series where True indicates an outlier.
    """
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    result = pd.Series(False, index=df.index)
    non_null_mask = df[column].notna()
    result[non_null_mask] = z_scores > threshold

    return result


def treat_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    action: str = "cap",
    multiplier: float = 1.5,
    zscore_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect and treat outliers in a numeric column.

    Parameters:
        df: Input DataFrame
        column: Column to process
        method: Detection method — 'iqr' or 'zscore'
        action: Treatment action — 'cap' (winsorize), 'remove', or 'nan'
        multiplier: IQR multiplier (for IQR method)
        zscore_threshold: Z-score threshold (for Z-score method)
    """
    df = df.copy()

    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        outlier_mask = detect_outliers_iqr(df, column, multiplier)
    elif method == "zscore":
        outlier_mask = detect_outliers_zscore(df, column, zscore_threshold)
        lower = df[column].mean() - zscore_threshold * df[column].std()
        upper = df[column].mean() + zscore_threshold * df[column].std()
    else:
        raise ValueError(f"Unknown method: {method}")

    outlier_count = outlier_mask.sum()

    if action == "cap":
        df[column] = df[column].clip(lower=lower, upper=upper)
    elif action == "remove":
        df = df[~outlier_mask]
    elif action == "nan":
        df.loc[outlier_mask, column] = np.nan
    else:
        raise ValueError(f"Unknown action: {action}")

    print(f"  → {column}: {outlier_count} outliers detected ({method}), action: {action}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DUPLICATE HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def find_duplicates(
    df: pd.DataFrame,
    subset: Optional[list] = None,
    keep: str = "first"
) -> pd.DataFrame:
    """
    Find duplicate rows in the DataFrame.

    Parameters:
        df: Input DataFrame
        subset: Columns to check for duplicates (None = all columns)
        keep: Which duplicates to mark — 'first', 'last', or False (mark all)

    Returns:
        DataFrame containing only the duplicate rows.
    """
    mask = df.duplicated(subset=subset, keep=keep)
    return df[mask]


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[list] = None,
    keep: str = "first"
) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    initial_count = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    removed = initial_count - len(df)
    print(f"  → Removed {removed:,} duplicate rows (checked: {subset or 'all columns'})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA STANDARDIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def standardize_text(
    df: pd.DataFrame,
    column: str,
    case: str = "title",
    strip: bool = True
) -> pd.DataFrame:
    """
    Standardize text formatting in a column.

    Parameters:
        case: 'title', 'lower', 'upper', or 'sentence'
        strip: Whether to strip leading/trailing whitespace
    """
    df = df.copy()
    col = df[column].astype(str)

    if strip:
        col = col.str.strip()

    # Remove extra internal whitespace
    col = col.str.replace(r"\s+", " ", regex=True)

    if case == "title":
        col = col.str.title()
    elif case == "lower":
        col = col.str.lower()
    elif case == "upper":
        col = col.str.upper()
    elif case == "sentence":
        col = col.str.capitalize()

    # Restore NaN values
    col[col == "Nan"] = np.nan
    col[col == "nan"] = np.nan
    col[col == "NAN"] = np.nan

    df[column] = col
    return df


def standardize_categories(
    df: pd.DataFrame,
    column: str,
    mapping: dict
) -> pd.DataFrame:
    """
    Map messy category values to standardized names.

    Parameters:
        mapping: Dict mapping messy names to clean names.
                 Keys should be lowercase for case-insensitive matching.
    """
    df = df.copy()

    def clean_category(val):
        if pd.isna(val):
            return val
        cleaned = str(val).strip().lower()
        # Try exact match first
        if cleaned in mapping:
            return mapping[cleaned]
        # Try partial/fuzzy match
        for key, value in mapping.items():
            if cleaned.startswith(key[:4]):
                return value
        return val  # Return original if no match

    df[column] = df[column].apply(clean_category)
    return df


def parse_dates(
    df: pd.DataFrame,
    column: str,
    target_format: str = "%Y-%m-%d"
) -> pd.DataFrame:
    """
    Parse dates from various formats into a standardized format.
    Uses pandas' flexible date parser with coercion for unparseable values.
    """
    df = df.copy()
    df[column] = pd.to_datetime(df[column], format="mixed", dayfirst=False, errors="coerce")
    return df


def clean_numeric_column(
    df: pd.DataFrame,
    column: str,
    remove_chars: Optional[list] = None
) -> pd.DataFrame:
    """
    Clean a numeric column by removing non-numeric characters and converting to float.

    Parameters:
        remove_chars: Characters to strip before conversion (e.g., ['₹', '$', '€', ','])
    """
    df = df.copy()

    def clean_value(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val)
        if remove_chars:
            for char in remove_chars:
                val_str = val_str.replace(char, "")
        try:
            return float(val_str)
        except ValueError:
            return np.nan

    df[column] = df[column].apply(clean_value)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_range(
    df: pd.DataFrame,
    column: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    action: str = "nan"
) -> pd.DataFrame:
    """
    Validate that numeric values fall within an acceptable range.

    Parameters:
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
        action: What to do with invalid values — 'nan', 'drop', 'cap'
    """
    df = df.copy()
    invalid_mask = pd.Series(False, index=df.index)

    if min_val is not None:
        invalid_mask |= df[column] < min_val
    if max_val is not None:
        invalid_mask |= df[column] > max_val

    invalid_count = invalid_mask.sum()

    if action == "nan":
        df.loc[invalid_mask, column] = np.nan
    elif action == "drop":
        df = df[~invalid_mask]
    elif action == "cap":
        if min_val is not None:
            df[column] = df[column].clip(lower=min_val)
        if max_val is not None:
            df[column] = df[column].clip(upper=max_val)

    print(f"  → {column}: {invalid_count} invalid values (range: [{min_val}, {max_val}]), action: {action}")
    return df


def validate_email(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Validate email format and mark invalid ones as NaN."""
    df = df.copy()
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    def check_email(val):
        if pd.isna(val) or val == "":
            return np.nan
        if re.match(email_pattern, str(val)):
            return val
        return np.nan

    invalid_count = df[column].apply(
        lambda x: not bool(re.match(email_pattern, str(x))) if pd.notna(x) and x != "" else False
    ).sum()

    df[column] = df[column].apply(check_email)
    print(f"  → {column}: {invalid_count} invalid emails marked as NaN")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add commonly useful derived features from existing columns.
    Specific to the e-commerce dataset.
    """
    df = df.copy()

    # Total order amount
    if "quantity" in df.columns and "unit_price" in df.columns:
        df["total_amount"] = df["quantity"] * df["unit_price"]

        # Apply discount
        if "discount_pct" in df.columns:
            df["discount_amount"] = df["total_amount"] * (df["discount_pct"] / 100)
            df["final_amount"] = df["total_amount"] - df["discount_amount"]

    # Date-based features
    if "order_date" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["order_date"]):
            df["order_month"] = df["order_date"].dt.month
            df["order_quarter"] = df["order_date"].dt.quarter
            df["order_year"] = df["order_date"].dt.year
            df["order_day_of_week"] = df["order_date"].dt.day_name()
            df["is_weekend"] = df["order_date"].dt.dayofweek.isin([5, 6])

    # Age group
    if "customer_age" in df.columns:
        bins = [0, 25, 35, 45, 55, 100]
        labels = ["18-25", "26-35", "36-45", "46-55", "55+"]
        df["age_group"] = pd.cut(df["customer_age"], bins=bins, labels=labels)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CLEANING REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cleaning_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame
) -> str:
    """Generate a summary report comparing data before and after cleaning."""

    report = []
    report.append("=" * 60)
    report.append("  DATA CLEANING REPORT")
    report.append("=" * 60)
    report.append("")
    report.append(f"  Records before cleaning: {len(df_before):,}")
    report.append(f"  Records after cleaning:  {len(df_after):,}")
    report.append(f"  Records removed:         {len(df_before) - len(df_after):,}")
    report.append("")
    report.append("  Missing Values:")
    report.append(f"    Before: {df_before.isnull().sum().sum():,}")
    report.append(f"    After:  {df_after.isnull().sum().sum():,}")
    report.append("")
    report.append("  Duplicates:")
    report.append(f"    Before: {df_before.duplicated().sum():,}")
    report.append(f"    After:  {df_after.duplicated().sum():,}")
    report.append("")
    report.append("  Column count:")
    report.append(f"    Before: {len(df_before.columns)}")
    report.append(f"    After:  {len(df_after.columns)}")
    report.append("")
    report.append("=" * 60)

    return "\n".join(report)
