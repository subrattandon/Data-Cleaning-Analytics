"""
=============================================================================
Script 03: Professional Visualizations
=============================================================================
Purpose:
    Create publication-quality visualizations from the cleaned dataset.
    Demonstrates various chart types and visualization techniques using
    Matplotlib and Seaborn.

Charts Created:
    1. KPI Summary Dashboard
    2. Revenue Trend (Time Series)
    3. Category Performance (Bar Chart)
    4. Customer Age Distribution (Histogram + Violin)
    5. Correlation Heatmap
    6. Payment Method Analysis (Donut Chart)
    7. Shipping Performance (Box Plot)
    8. Rating Distribution
    9. Revenue by City (Horizontal Bar)
    10. Day-of-Week Patterns
=============================================================================
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

from src.viz_utils import (
    setup_style, save_figure, COLORS, CATEGORY_PALETTE,
    add_value_labels, format_currency_axis, create_kpi_card,
    add_watermark, styled_heatmap, format_indian_currency, format_indian_number
)

setup_style()

# ── Load Cleaned Data ──────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
CLEANED_DATA = os.path.join(PROJECT_DIR, "data", "cleaned", "ecommerce_sales_cleaned.csv")


def load_data():
    """Load the cleaned dataset."""
    if not os.path.exists(CLEANED_DATA):
        print("❌ Cleaned dataset not found!")
        print("   Run '02_data_cleaning.py' first.")
        sys.exit(1)

    df = pd.read_csv(CLEANED_DATA, parse_dates=["order_date"])
    print(f"✅ Loaded cleaned dataset: {len(df):,} rows × {len(df.columns)} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1: KPI Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def chart_kpi_dashboard(df):
    """Create a KPI summary dashboard with key metrics."""
    print("  📊 Creating KPI Dashboard...")

    fig = plt.figure(figsize=(20, 5))
    fig.patch.set_facecolor(COLORS["bg"])

    total_revenue = df["final_amount"].sum() if "final_amount" in df.columns else df["total_amount"].sum()
    total_orders = len(df)
    avg_order_val = total_revenue / total_orders if total_orders > 0 else 0
    avg_rating = df["rating"].mean()
    avg_shipping = df["shipping_days"].mean()

    kpis = [
        {"value": format_indian_currency(total_revenue), "label": "Total Revenue", "color": COLORS["primary"]},
        {"value": format_indian_number(total_orders), "label": "Total Orders", "color": COLORS["accent"]},
        {"value": format_indian_currency(avg_order_val, decimals=2), "label": "Avg Order Value", "color": COLORS["success"]},
        {"value": f"{avg_rating:.1f} ★", "label": "Avg Rating", "color": COLORS["warning"]},
        {"value": f"{avg_shipping:.1f} days", "label": "Avg Shipping", "color": COLORS["info"]},
    ]

    for i, kpi in enumerate(kpis):
        pos = [0.02 + i * 0.196, 0.1, 0.18, 0.8]
        create_kpi_card(fig, pos, kpi["value"], kpi["label"], color=kpi["color"])

    fig.suptitle("E-Commerce Sales — Key Performance Indicators",
                 fontsize=16, fontweight="bold", color=COLORS["text"], y=1.08)
    add_watermark(fig)
    save_figure(fig, "03_kpi_dashboard.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2: Revenue Trend (Time Series)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_revenue_trend(df):
    """Create a time series chart of monthly revenue."""
    print("  📊 Creating Revenue Trend...")

    amount_col = "final_amount" if "final_amount" in df.columns else "total_amount"

    monthly = df.set_index("order_date").resample("ME")[amount_col].sum().reset_index()
    monthly.columns = ["month", "revenue"]

    fig, ax = plt.subplots(figsize=(16, 7))

    # Area chart with gradient
    ax.fill_between(monthly["month"], monthly["revenue"],
                    alpha=0.15, color=COLORS["primary"])
    ax.plot(monthly["month"], monthly["revenue"],
            color=COLORS["primary"], linewidth=2.5, marker="o",
            markersize=6, markerfacecolor=COLORS["accent"],
            markeredgecolor=COLORS["bg"], markeredgewidth=2)

    # Highlight peak month
    peak_idx = monthly["revenue"].idxmax()
    peak_month = monthly.loc[peak_idx, "month"]
    peak_revenue = monthly.loc[peak_idx, "revenue"]
    ax.annotate(
        f"Peak: {format_indian_currency(peak_revenue)}",
        xy=(peak_month, peak_revenue),
        xytext=(30, 30), textcoords="offset points",
        fontsize=11, fontweight="bold", color=COLORS["accent"],
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["card_bg"],
                  edgecolor=COLORS["accent"], alpha=0.9)
    )

    # Add trend line
    x_numeric = np.arange(len(monthly))
    z = np.polyfit(x_numeric, monthly["revenue"].values, 1)
    p = np.poly1d(z)
    ax.plot(monthly["month"], p(x_numeric), "--",
            color=COLORS["warning"], alpha=0.6, linewidth=1.5,
            label="Trend Line")

    format_currency_axis(ax, "y")
    ax.set_title("Monthly Revenue Trend", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.legend(loc="upper left")
    add_watermark(fig)

    plt.tight_layout()
    save_figure(fig, "03_revenue_trend.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3: Category Performance
# ═══════════════════════════════════════════════════════════════════════════════

def chart_category_performance(df):
    """Create a grouped bar chart comparing categories."""
    print("  📊 Creating Category Performance...")

    amount_col = "final_amount" if "final_amount" in df.columns else "total_amount"

    cat_stats = df.groupby("category").agg(
        total_revenue=(amount_col, "sum"),
        avg_order_value=(amount_col, "mean"),
        order_count=(amount_col, "count"),
        avg_rating=("rating", "mean")
    ).sort_values("total_revenue", ascending=True).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: Revenue by category
    ax = axes[0]
    colors = [CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]
              for i in range(len(cat_stats))]
    bars = ax.barh(cat_stats["category"], cat_stats["total_revenue"],
                   color=colors, alpha=0.85, edgecolor=COLORS["grid"],
                   linewidth=0.5)

    for bar, val in zip(bars, cat_stats["total_revenue"]):
        label = format_indian_currency(val, is_kpi=True)
        ax.text(bar.get_width() + bar.get_width() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10,
                color=COLORS["text_secondary"], fontweight="bold")

    ax.set_title("Total Revenue by Category", fontsize=14, fontweight="bold", pad=12)
    format_currency_axis(ax, "x")

    # Right: Avg rating by category
    ax = axes[1]
    bars = ax.barh(cat_stats["category"], cat_stats["avg_rating"],
                   color=colors, alpha=0.85, edgecolor=COLORS["grid"],
                   linewidth=0.5)

    for bar, val in zip(bars, cat_stats["avg_rating"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} ★", va="center", fontsize=10,
                color=COLORS["text_secondary"], fontweight="bold")

    ax.set_title("Average Rating by Category", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, 5.5)

    fig.suptitle("Category Performance Analysis", fontsize=17,
                 fontweight="bold", color=COLORS["primary"], y=1.02)
    add_watermark(fig)
    plt.tight_layout()
    save_figure(fig, "03_category_performance.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4: Customer Demographics
# ═══════════════════════════════════════════════════════════════════════════════

def chart_customer_demographics(df):
    """Create visualizations of customer age demographics."""
    print("  📊 Creating Customer Demographics...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Age distribution histogram
    ax = axes[0]
    sns.histplot(df["customer_age"].dropna(), bins=30, kde=True,
                 ax=ax, color=COLORS["primary"], alpha=0.6,
                 edgecolor=COLORS["grid"], linewidth=0.5)

    mean_age = df["customer_age"].mean()
    ax.axvline(mean_age, color=COLORS["accent"], linestyle="--",
               linewidth=2, label=f"Mean: {mean_age:.0f} years")
    ax.set_title("Customer Age Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age")
    ax.legend()

    # Right: Age group vs revenue
    ax = axes[1]
    if "age_group" in df.columns:
        amount_col = "final_amount" if "final_amount" in df.columns else "total_amount"
        age_revenue = df.groupby("age_group", observed=True)[amount_col].mean().sort_index()

        bars = ax.bar(age_revenue.index.astype(str), age_revenue.values,
                      color=CATEGORY_PALETTE[:len(age_revenue)], alpha=0.85,
                      edgecolor=COLORS["grid"], linewidth=0.5)

        for bar, val in zip(bars, age_revenue.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    format_indian_currency(val), ha="center", fontsize=10,
                    color=COLORS["text_secondary"], fontweight="bold")

        ax.set_title("Avg Order Value by Age Group", fontsize=14, fontweight="bold")
        ax.set_xlabel("Age Group")
        format_currency_axis(ax, "y")

    fig.suptitle("Customer Demographics Analysis", fontsize=17,
                 fontweight="bold", color=COLORS["primary"], y=1.02)
    add_watermark(fig)
    plt.tight_layout()
    save_figure(fig, "03_customer_demographics.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5: Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def chart_correlation_heatmap(df):
    """Create a correlation heatmap of numeric variables."""
    print("  📊 Creating Correlation Heatmap...")

    numeric_cols = ["quantity", "unit_price", "discount_pct",
                    "shipping_days", "rating", "customer_age"]

    # Add derived columns if available
    for col in ["total_amount", "final_amount"]:
        if col in df.columns:
            numeric_cols.append(col)

    available_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[available_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    styled_heatmap(corr, "Feature Correlation Matrix", ax=ax)
    add_watermark(fig)
    plt.tight_layout()
    save_figure(fig, "03_correlation_heatmap.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6: Payment Method Analysis (Donut)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_payment_methods(df):
    """Create a donut chart of payment method distribution."""
    print("  📊 Creating Payment Method Analysis...")

    payment_counts = df["payment_method"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Donut chart
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        payment_counts.values, labels=payment_counts.index,
        autopct="%1.1f%%", colors=CATEGORY_PALETTE[:len(payment_counts)],
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor=COLORS["bg"], linewidth=2),
        textprops=dict(color=COLORS["text"], fontsize=10)
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)
        autotext.set_color(COLORS["text"])

    ax.set_title("Payment Method Distribution", fontsize=14, fontweight="bold")

    # Center text
    ax.text(0, 0, f"{format_indian_number(len(df))}\nOrders", ha="center", va="center",
            fontsize=14, fontweight="bold", color=COLORS["text"])

    # Right: Avg order value by payment method
    ax = axes[1]
    amount_col = "final_amount" if "final_amount" in df.columns else "total_amount"
    payment_aov = df.groupby("payment_method")[amount_col].mean().sort_values()

    bars = ax.barh(payment_aov.index, payment_aov.values,
                   color=CATEGORY_PALETTE[:len(payment_aov)], alpha=0.85,
                   edgecolor=COLORS["grid"], linewidth=0.5)

    for bar, val in zip(bars, payment_aov.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                format_indian_currency(val, decimals=2), va="center", fontsize=10,
                color=COLORS["text_secondary"], fontweight="bold")

    ax.set_title("Avg Order Value by Payment Method",
                 fontsize=14, fontweight="bold")
    format_currency_axis(ax, "x")

    fig.suptitle("Payment Analysis", fontsize=17,
                 fontweight="bold", color=COLORS["primary"], y=1.02)
    add_watermark(fig)
    plt.tight_layout()
    save_figure(fig, "03_payment_analysis.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7: Shipping Performance (Box Plot)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_shipping_performance(df):
    """Create box plots for shipping days by category."""
    print("  📊 Creating Shipping Performance...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Box plot of shipping days by category
    ax = axes[0]
    order = df.groupby("category")["shipping_days"].median().sort_values().index
    sns.boxplot(data=df, x="category", y="shipping_days", order=order,
                ax=ax, palette=CATEGORY_PALETTE, flierprops=dict(
                    markerfacecolor=COLORS["accent"], markersize=4, alpha=0.5
                ))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Shipping Days by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("")

    # Right: Violin plot of rating by category
    ax = axes[1]
    sns.violinplot(data=df, x="category", y="rating", order=order,
                   ax=ax, palette=CATEGORY_PALETTE, inner="box", alpha=0.7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Rating Distribution by Category",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("")

    fig.suptitle("Shipping & Rating Performance", fontsize=17,
                 fontweight="bold", color=COLORS["primary"], y=1.02)
    add_watermark(fig)
    plt.tight_layout()
    save_figure(fig, "03_shipping_rating.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 8: Day of Week Patterns
# ═══════════════════════════════════════════════════════════════════════════════

def chart_day_patterns(df):
    """Analyze sales patterns by day of week."""
    print("  📊 Creating Day-of-Week Patterns...")

    if "order_day_of_week" not in df.columns:
        print("    ⚠️  order_day_of_week not found. Skipping.")
        return

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

    amount_col = "final_amount" if "final_amount" in df.columns else "total_amount"

    day_stats = df.groupby("order_day_of_week").agg(
        order_count=("order_id", "count"),
        total_revenue=(amount_col, "sum"),
        avg_revenue=(amount_col, "mean")
    ).reindex(day_order)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Order count by day
    ax = axes[0]
    colors = [COLORS["accent"] if day in ["Saturday", "Sunday"]
              else COLORS["primary"] for day in day_order]
    bars = ax.bar(day_stats.index, day_stats["order_count"],
                  color=colors, alpha=0.85, edgecolor=COLORS["grid"])
    ax.set_title("Order Count by Day of Week", fontsize=14, fontweight="bold")
    ax.set_xticklabels(day_order, rotation=45, ha="right")

    for bar, val in zip(bars, day_stats["order_count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                format_indian_number(val), ha="center", fontsize=9,
                color=COLORS["text_secondary"], fontweight="bold")

    # Right: Avg revenue by day
    ax = axes[1]
    bars = ax.bar(day_stats.index, day_stats["avg_revenue"],
                  color=colors, alpha=0.85, edgecolor=COLORS["grid"])
    ax.set_title("Average Order Value by Day", fontsize=14, fontweight="bold")
    ax.set_xticklabels(day_order, rotation=45, ha="right")
    format_currency_axis(ax, "y")

    # Highlight weekends
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["primary"], alpha=0.85, label="Weekday"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["accent"], alpha=0.85, label="Weekend"),
    ], loc="upper right")

    fig.suptitle("Weekly Sales Patterns", fontsize=17,
                 fontweight="bold", color=COLORS["primary"], y=1.02)
    add_watermark(fig)
    plt.tight_layout()
    save_figure(fig, "03_day_patterns.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 9: Top Cities
# ═══════════════════════════════════════════════════════════════════════════════

def chart_top_cities(df):
    """Visualize revenue by city."""
    print("  📊 Creating Top Cities Chart...")

    amount_col = "final_amount" if "final_amount" in df.columns else "total_amount"

    city_revenue = df.groupby("city")[amount_col].sum().sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create gradient colors
    n = len(city_revenue)
    gradient_colors = plt.cm.viridis(np.linspace(0.3, 0.9, n))

    bars = ax.barh(city_revenue.index, city_revenue.values,
                   color=gradient_colors, edgecolor=COLORS["grid"],
                   linewidth=0.5, alpha=0.9)

    for bar, val in zip(bars, city_revenue.values):
        label = format_indian_currency(val, is_kpi=True)
        ax.text(bar.get_width() + bar.get_width() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10,
                color=COLORS["text_secondary"], fontweight="bold")

    ax.set_title("Top 15 Cities by Revenue", fontsize=16, fontweight="bold", pad=15)
    format_currency_axis(ax, "x")
    add_watermark(fig)
    plt.tight_layout()
    save_figure(fig, "03_top_cities.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 10: Scatter — Price vs Rating
# ═══════════════════════════════════════════════════════════════════════════════

def chart_price_vs_rating(df):
    """Create a scatter plot of price vs rating, colored by category."""
    print("  📊 Creating Price vs Rating Scatter...")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Sample for readability
    sample = df.sample(min(2000, len(df)), random_state=42)

    categories = sample["category"].unique()
    for i, cat in enumerate(categories):
        cat_data = sample[sample["category"] == cat]
        ax.scatter(cat_data["unit_price"], cat_data["rating"],
                   c=CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)],
                   label=cat, alpha=0.5, s=30, edgecolors="none")

    ax.set_title("Unit Price vs Customer Rating",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Unit Price (₹)")
    ax.set_ylabel("Rating (1-5)")
    ax.legend(loc="lower right", fontsize=9, ncol=2,
              framealpha=0.8)
    format_currency_axis(ax, "x")
    add_watermark(fig)

    plt.tight_layout()
    save_figure(fig, "03_price_vs_rating.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  🎨 PROFESSIONAL VISUALIZATIONS")
    print("  E-Commerce Sales Analytics")
    print("=" * 70 + "\n")

    df = load_data()

    # Generate all charts
    chart_kpi_dashboard(df)
    chart_revenue_trend(df)
    chart_category_performance(df)
    chart_customer_demographics(df)
    chart_correlation_heatmap(df)
    chart_payment_methods(df)
    chart_shipping_performance(df)
    chart_day_patterns(df)
    chart_top_cities(df)
    chart_price_vs_rating(df)

    print("\n" + "=" * 70)
    print("  ✅ ALL VISUALIZATIONS CREATED")
    print("  📁 Check the reports/ directory for saved charts")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
