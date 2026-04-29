"""
=============================================================================
Visualization Utilities
=============================================================================
Helper functions for creating professional, publication-quality charts
using Matplotlib and Seaborn. Provides consistent styling across all
visualizations in the project.
=============================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
import os

# ── Global Style Configuration ─────────────────────────────────────────────────

# Premium color palette
COLORS = {
    "primary": "#6366F1",      # Indigo
    "secondary": "#8B5CF6",    # Violet
    "accent": "#EC4899",       # Pink
    "success": "#10B981",      # Emerald
    "warning": "#F59E0B",      # Amber
    "danger": "#EF4444",       # Red
    "info": "#3B82F6",         # Blue
    "dark": "#1E1B4B",         # Dark indigo
    "light": "#F8FAFC",        # Slate 50
    "muted": "#94A3B8",        # Slate 400
    "bg": "#0F172A",           # Slate 900
    "card_bg": "#1E293B",      # Slate 800
    "grid": "#334155",         # Slate 700
    "text": "#F1F5F9",         # Slate 100
    "text_secondary": "#CBD5E1" # Slate 300
}

# Categorical palette (colorblind-friendly)
CATEGORY_PALETTE = [
    "#6366F1", "#EC4899", "#10B981", "#F59E0B",
    "#3B82F6", "#8B5CF6", "#14B8A6", "#F97316"
]

# Sequential palette
SEQUENTIAL_PALETTE = "viridis"

# Diverging palette
DIVERGING_PALETTE = "coolwarm"


def setup_style():
    """Apply premium dark-mode styling to all matplotlib plots."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        # Figure
        "figure.facecolor": COLORS["bg"],
        "figure.edgecolor": COLORS["bg"],
        "figure.figsize": (12, 7),
        "figure.dpi": 150,

        # Axes
        "axes.facecolor": COLORS["card_bg"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "axes.titlecolor": COLORS["text"],
        "axes.labelsize": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Grid
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,

        # Text
        "text.color": COLORS["text"],
        "font.family": "sans-serif",
        "font.size": 11,

        # Ticks
        "xtick.color": COLORS["text_secondary"],
        "ytick.color": COLORS["text_secondary"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Legend
        "legend.facecolor": COLORS["card_bg"],
        "legend.edgecolor": COLORS["grid"],
        "legend.fontsize": 10,

        # Save
        "savefig.facecolor": COLORS["bg"],
        "savefig.edgecolor": COLORS["bg"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


def get_reports_dir():
    """Get the reports directory path and create it if needed."""
    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "reports"
    )
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir


def save_figure(fig, filename: str, dpi: int = 200):
    """Save a figure to the reports directory."""
    reports_dir = get_reports_dir()
    filepath = os.path.join(reports_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    print(f"  💾 Saved: {filepath}")
    return filepath


def add_value_labels(ax, fmt="{:.0f}", fontsize=9, color=None, offset=5):
    """Add value labels on top of bar chart bars."""
    label_color = color or COLORS["text_secondary"]
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + offset,
                fmt.format(height),
                ha="center", va="bottom",
                fontsize=fontsize, color=label_color,
                fontweight="bold"
            )


def format_indian_currency(num, is_kpi=False, decimals=0):
    """Format a number into Indian currency format (e.g., ₹10,00,000 or ₹1.5 Cr)."""
    if pd.isna(num): return "₹0"
    
    if is_kpi:
        if abs(num) >= 1_00_00_000:
            return f"₹{num/1_00_00_000:.2f} Cr"
        elif abs(num) >= 1_00_000:
            return f"₹{num/1_00_000:.2f} L"
        elif abs(num) >= 1_000:
            return f"₹{num/1_000:.1f} K"
            
    is_negative = num < 0
    num = abs(num)
    
    if decimals > 0:
        int_part = int(num)
        dec_part = f"{num:.{decimals}f}".split('.')[1]
    else:
        int_part = int(round(num))
        dec_part = ""
        
    s = str(int_part)
    if len(s) <= 3:
        formatted_int = s
    else:
        last_three = s[-3:]
        other_digits = s[:-3]
        parts = []
        while len(other_digits) > 0:
            parts.insert(0, other_digits[-2:])
            other_digits = other_digits[:-2]
        formatted_int = f"{','.join(parts)},{last_three}"
        
    res = f"₹{formatted_int}"
    if decimals > 0:
        res += f".{dec_part}"
        
    return f"-{res}" if is_negative else res

def format_indian_number(num):
    """Format a normal number into Indian comma format (e.g., 10,00,000)."""
    if pd.isna(num): return "0"
    is_negative = num < 0
    num = abs(int(num))
    s = str(num)
    if len(s) <= 3:
        formatted = s
    else:
        last_three = s[-3:]
        other_digits = s[:-3]
        parts = []
        while len(other_digits) > 0:
            parts.insert(0, other_digits[-2:])
            other_digits = other_digits[:-2]
        formatted = f"{','.join(parts)},{last_three}"
    return f"-{formatted}" if is_negative else formatted

def format_currency_axis(ax, axis="y"):
    """Format an axis to show currency values (e.g., ₹1.2L, ₹3.5Cr)."""
    def currency_formatter(x, p):
        if abs(x) >= 1_00_00_000:
            return f"₹{x/1_00_00_000:.1f}Cr"
        elif abs(x) >= 1_00_000:
            return f"₹{x/1_00_000:.1f}L"
        elif abs(x) >= 1_000:
            return f"₹{x/1_000:.1f}K"
        else:
            return f"₹{x:.0f}"

    if axis == "y":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(currency_formatter))
    else:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(currency_formatter))


def create_kpi_card(fig, position, value, label, delta=None, color=None):
    """
    Create a KPI card within a figure.

    Parameters:
        fig: matplotlib Figure
        position: [left, bottom, width, height] in figure coordinates
        value: The main KPI value (string)
        label: The KPI label
        delta: Optional change indicator (e.g., "+12%")
        color: KPI accent color
    """
    card_color = color or COLORS["primary"]
    ax = fig.add_axes(position)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(COLORS["card_bg"])
    ax.axis("off")

    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(card_color)
        spine.set_linewidth(2)

    # Value
    ax.text(0.5, 0.6, value, transform=ax.transAxes,
            fontsize=28, fontweight="bold", color=card_color,
            ha="center", va="center")

    # Label
    ax.text(0.5, 0.25, label, transform=ax.transAxes,
            fontsize=11, color=COLORS["text_secondary"],
            ha="center", va="center")

    # Delta
    if delta:
        delta_color = COLORS["success"] if delta.startswith("+") else COLORS["danger"]
        ax.text(0.5, 0.85, delta, transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=delta_color,
                ha="center", va="center")

    return ax


def add_watermark(fig, text="E-Commerce Analytics"):
    """Add a subtle watermark to the figure."""
    fig.text(
        0.99, 0.01, text,
        fontsize=8, color=COLORS["muted"],
        ha="right", va="bottom", alpha=0.4,
        style="italic"
    )


def styled_heatmap(data, title, ax=None, fmt=".2f", cmap=None):
    """Create a styled correlation heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(data, dtype=bool), k=1)
    cmap = cmap or DIVERGING_PALETTE

    sns.heatmap(
        data, mask=mask, annot=True, fmt=fmt,
        cmap=cmap, center=0, ax=ax,
        linewidths=0.5, linecolor=COLORS["grid"],
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        annot_kws={"fontsize": 9}
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    return ax
