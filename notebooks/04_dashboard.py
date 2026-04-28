"""
=============================================================================
Script 04: Interactive Streamlit Dashboard
=============================================================================
Purpose:
    An interactive dashboard for exploring cleaned e-commerce sales data.
    Features filters, KPI cards, and dynamic charts.

Run:
    streamlit run notebooks/04_dashboard.py
=============================================================================
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0F172A;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #1E1B4B 100%);
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #334155;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #F1F5F9;
        padding: 8px 0 16px 0;
        border-bottom: 2px solid #6366F1;
        margin-bottom: 16px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Plotly Theme ───────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#0F172A",
        "plot_bgcolor": "#1E293B",
        "font": {"color": "#F1F5F9", "family": "Inter, sans-serif"},
        "xaxis": {"gridcolor": "#334155", "linecolor": "#334155"},
        "yaxis": {"gridcolor": "#334155", "linecolor": "#334155"},
        "colorway": ["#6366F1", "#EC4899", "#10B981", "#F59E0B",
                      "#3B82F6", "#8B5CF6", "#14B8A6", "#F97316"],
        "margin": {"l": 40, "r": 20, "t": 50, "b": 40},
    }
}

COLORS = {
    "primary": "#6366F1",
    "accent": "#EC4899",
    "success": "#10B981",
    "warning": "#F59E0B",
    "info": "#3B82F6",
}


# ── Data Loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load and cache the cleaned dataset."""
    project_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_dir, "data", "cleaned",
                             "ecommerce_sales_cleaned.csv")

    if not os.path.exists(data_path):
        st.error("❌ Cleaned dataset not found! Run `02_data_cleaning.py` first.")
        st.stop()

    df = pd.read_csv(data_path, parse_dates=["order_date"])
    return df


# ── Helper Functions ───────────────────────────────────────────────────────────

def kpi_card(label, value, color="#6366F1"):
    """Render a KPI card."""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(text):
    """Render a section header."""
    st.markdown(f'<div class="section-header">{text}</div>',
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Load data
    df = load_data()

    # ── Sidebar Filters ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("# 🎛️ Filters")
        st.markdown("---")

        # Date range filter
        if df["order_date"].notna().any():
            min_date = df["order_date"].min().date()
            max_date = df["order_date"].max().date()
            date_range = st.date_input(
                "📅 Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        else:
            date_range = None

        # Category filter
        categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
        selected_category = st.selectbox("🏷️ Category", categories)

        # City filter
        cities = ["All"] + sorted(df["city"].dropna().unique().tolist())
        selected_city = st.selectbox("🌆 City", cities)

        # Payment method filter
        payments = ["All"] + sorted(df["payment_method"].dropna().unique().tolist())
        selected_payment = st.selectbox("💳 Payment Method", payments)

        # Rating filter
        min_rating = float(df["rating"].min()) if df["rating"].notna().any() else 1.0
        max_rating = float(df["rating"].max()) if df["rating"].notna().any() else 5.0
        rating_range = st.slider("⭐ Rating Range", min_rating, max_rating,
                                 (min_rating, max_rating), 0.1)

        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.info(f"**{len(df):,}** total records\n\n"
                f"**{len(df.columns)}** columns\n\n"
                f"**{df['category'].nunique()}** categories")

    # ── Apply Filters ──────────────────────────────────────────────────────
    filtered = df.copy()

    if date_range and len(date_range) == 2:
        filtered = filtered[
            (filtered["order_date"].dt.date >= date_range[0]) &
            (filtered["order_date"].dt.date <= date_range[1])
        ]

    if selected_category != "All":
        filtered = filtered[filtered["category"] == selected_category]

    if selected_city != "All":
        filtered = filtered[filtered["city"] == selected_city]

    if selected_payment != "All":
        filtered = filtered[filtered["payment_method"] == selected_payment]

    filtered = filtered[
        (filtered["rating"] >= rating_range[0]) &
        (filtered["rating"] <= rating_range[1])
    ]

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style='text-align: center; color: #F1F5F9; margin-bottom: 0;'>
        📊 E-Commerce Sales Dashboard
    </h1>
    <p style='text-align: center; color: #94A3B8; font-size: 1.1rem;'>
        Interactive analytics for cleaned e-commerce sales data
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"<p style='text-align: center; color: #64748B;'>"
                f"Showing <b>{len(filtered):,}</b> of {len(df):,} records "
                f"({len(filtered)/len(df)*100:.1f}%)</p>",
                unsafe_allow_html=True)

    # ── KPI Cards ──────────────────────────────────────────────────────────
    amount_col = "final_amount" if "final_amount" in filtered.columns else "total_amount"

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_rev = filtered[amount_col].sum() if amount_col in filtered.columns else 0
        if total_rev >= 1_000_000:
            rev_str = f"₹{total_rev/1_000_000:.2f}M"
        elif total_rev >= 1_000:
            rev_str = f"₹{total_rev/1_000:.1f}K"
        else:
            rev_str = f"₹{total_rev:,.0f}"
        kpi_card("Total Revenue", rev_str, COLORS["primary"])

    with col2:
        kpi_card("Total Orders", f"{len(filtered):,}", COLORS["accent"])

    with col3:
        aov = filtered[amount_col].mean() if amount_col in filtered.columns and len(filtered) > 0 else 0
        kpi_card("Avg Order Value", f"₹{aov:,.2f}", COLORS["success"])

    with col4:
        avg_r = filtered["rating"].mean() if len(filtered) > 0 else 0
        kpi_card("Avg Rating", f"{avg_r:.2f} ★", COLORS["warning"])

    with col5:
        avg_ship = filtered["shipping_days"].mean() if len(filtered) > 0 else 0
        kpi_card("Avg Ship Days", f"{avg_ship:.1f}", COLORS["info"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Revenue Trend + Category Breakdown ──────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        section_header("📈 Revenue Trend")

        if len(filtered) > 0 and amount_col in filtered.columns:
            monthly = (filtered.set_index("order_date")
                       .resample("ME")[amount_col].sum().reset_index())
            monthly.columns = ["month", "revenue"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly["month"], y=monthly["revenue"],
                mode="lines+markers",
                line=dict(color=COLORS["primary"], width=3),
                marker=dict(size=8, color=COLORS["accent"]),
                fill="tozeroy",
                fillcolor="rgba(99, 102, 241, 0.1)",
                name="Revenue"
            ))
            fig.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                height=380,
                xaxis_title="Month",
                yaxis_title="Revenue (₹)",
                showlegend=False,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("🏷️ Category Split")

        if len(filtered) > 0:
            cat_rev = filtered.groupby("category")[amount_col].sum().sort_values(ascending=False)

            fig = go.Figure(data=[go.Pie(
                labels=cat_rev.index, values=cat_rev.values,
                hole=0.55,
                textposition="inside",
                textinfo="label+percent",
                textfont=dict(size=10, color="#F1F5F9"),
                marker=dict(
                    colors=["#6366F1", "#EC4899", "#10B981", "#F59E0B",
                            "#3B82F6", "#8B5CF6", "#14B8A6", "#F97316"],
                    line=dict(color="#0F172A", width=2)
                )
            )])
            fig.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                height=380,
                showlegend=False,
                annotations=[dict(
                    text=f"₹{cat_rev.sum():,.0f}",
                    x=0.5, y=0.5, font_size=16,
                    font_color=COLORS["primary"],
                    showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Top Cities + Payment Methods ────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        section_header("🌆 Top 10 Cities by Revenue")

        if len(filtered) > 0:
            city_rev = (filtered.groupby("city")[amount_col].sum()
                        .sort_values(ascending=True).tail(10))

            fig = go.Figure(data=[go.Bar(
                y=city_rev.index, x=city_rev.values,
                orientation="h",
                marker=dict(
                    color=city_rev.values,
                    colorscale="Viridis",
                    line=dict(color="#334155", width=0.5)
                ),
                text=[f"₹{v:,.0f}" for v in city_rev.values],
                textposition="outside",
                textfont=dict(color="#CBD5E1", size=10)
            )])
            fig.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                height=400,
                xaxis_title="Revenue (₹)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("💳 Payment Method Analysis")

        if len(filtered) > 0:
            payment_stats = filtered.groupby("payment_method").agg(
                count=("order_id", "count"),
                avg_amount=(amount_col, "mean")
            ).sort_values("count", ascending=False)

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(go.Bar(
                x=payment_stats.index,
                y=payment_stats["count"],
                name="Order Count",
                marker_color=COLORS["primary"],
                opacity=0.8
            ), secondary_y=False)

            fig.add_trace(go.Scatter(
                x=payment_stats.index,
                y=payment_stats["avg_amount"],
                name="Avg Amount",
                mode="lines+markers",
                line=dict(color=COLORS["accent"], width=3),
                marker=dict(size=10)
            ), secondary_y=True)

            fig.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                height=400,
                legend=dict(x=0.02, y=0.98),
                hovermode="x unified"
            )
            fig.update_yaxes(title_text="Order Count", secondary_y=False)
            fig.update_yaxes(title_text="Avg Amount (₹)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Rating Distribution + Shipping ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        section_header("⭐ Rating Distribution")

        if len(filtered) > 0:
            fig = go.Figure(data=[go.Histogram(
                x=filtered["rating"].dropna(),
                nbinsx=20,
                marker=dict(
                    color=COLORS["warning"],
                    line=dict(color="#334155", width=1)
                ),
                opacity=0.85
            )])
            avg_rating = filtered["rating"].mean()
            fig.add_vline(x=avg_rating, line_dash="dash",
                         line_color=COLORS["accent"],
                         annotation_text=f"Avg: {avg_rating:.2f}")
            fig.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                height=380,
                xaxis_title="Rating",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("📦 Shipping Days Distribution")

        if len(filtered) > 0:
            fig = go.Figure(data=[go.Box(
                y=filtered["shipping_days"].dropna(),
                x=filtered["category"],
                marker_color=COLORS["info"],
                line=dict(color=COLORS["primary"]),
                fillcolor="rgba(99, 102, 241, 0.3)"
            )])
            fig.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                height=380,
                xaxis_title="Category",
                yaxis_title="Shipping Days",
                showlegend=False
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Data Table ──────────────────────────────────────────────────
    section_header("📋 Data Explorer")

    with st.expander("View Raw Data", expanded=False):
        display_cols = [c for c in filtered.columns if c not in [
            "discount_amount", "is_weekend", "order_month",
            "order_quarter", "order_year"
        ]]
        st.dataframe(
            filtered[display_cols].head(500),
            use_container_width=True,
            height=400
        )

    # ── Download Button ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        csv = filtered.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Filtered Data (CSV)",
            data=csv,
            file_name="ecommerce_filtered_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #64748B; font-size: 0.85rem;'>"
        "E-Commerce Sales Analytics Dashboard | Data Cleaning & Visualization Project | "
        "Built with Streamlit & Plotly</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
