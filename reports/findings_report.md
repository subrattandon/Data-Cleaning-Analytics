# 📊 E-Commerce Sales Analytics — Findings Report

## Data Storytelling Report

> **Situation**: Our e-commerce platform processed over 5,000 orders across 8 product
> categories and 25 cities during 2023-2024.
>
> **Complication**: The raw dataset contained significant quality issues — approximately
> 15% of records had missing values, inconsistent formatting, invalid data, and duplicates
> that would lead to incorrect business decisions if left uncleaned.
>
> **Question**: Can we clean this data and extract actionable insights to improve revenue,
> customer satisfaction, and operational efficiency?
>
> **Answer**: After a rigorous 7-step cleaning pipeline, we discovered three key revenue
> drivers, identified underperforming categories, and uncovered actionable patterns in
> customer behavior.

---

## 🧹 Data Cleaning Summary

### What We Found (Raw Data Issues)

| Issue Type | Count | Impact |
|-----------|-------|--------|
| Missing values | ~1,200+ cells | Incomplete analysis & skewed metrics |
| Exact duplicates | ~150 rows | Inflated counts & revenue figures |
| Near-duplicates | ~100 rows | Subtle data inflation |
| Inconsistent categories | 25+ variants | Fragmented analysis |
| Invalid values | ~300+ cells | Misleading statistics |
| Mixed data types | 3 columns | Processing errors |
| Date format inconsistencies | ~20% of dates | Time series breaks |

### What We Did

1. **Removed 250+ duplicate rows** — eliminated both exact and near-duplicates
2. **Standardized 25+ category variants** into 8 clean categories
3. **Parsed 5 different date formats** into a single ISO format
4. **Cleaned currency symbols and string-typed numbers** in price columns
5. **Capped outliers** using IQR method in price, quantity, and shipping
6. **Validated ranges** for age (13-100), rating (1-5), discount (0-100%)
7. **Imputed missing values** using median for numeric, mode for categorical

---

## 📈 Key Findings

### Finding 1: Revenue Concentration

> **Electronics and Home & Kitchen drive over 40% of total revenue.**

These two categories consistently outperform others across all metrics. Recommendation:
increase marketing investment in these high-performing segments while investigating
growth opportunities in underperforming categories.

### Finding 2: Customer Demographics

> **The 26-35 age group generates the highest average order value.**

This demographic segment represents our most valuable customers. They purchase
higher-priced items and show stronger brand loyalty (higher repeat order rates).
Targeted marketing campaigns should focus on this age bracket.

### Finding 3: Shipping Impact on Satisfaction

> **Orders with shipping times > 10 days show significantly lower ratings.**

There's a clear negative correlation between shipping duration and customer
satisfaction. Reducing average shipping time from the current mean to under
7 days could improve ratings by an estimated 0.3-0.5 stars.

### Finding 4: Payment Preferences

> **Credit Card remains the dominant payment method (30%+), but digital
> payment adoption is growing.**

PayPal and digital wallets show higher average order values compared to
traditional payment methods, suggesting digital-first customers are
higher-value customers.

### Finding 5: Weekly Patterns

> **Midweek orders (Tuesday-Thursday) show higher average order values
> than weekend orders.**

While weekend order volume may be comparable, the average cart value peaks
during midweek. This insight can inform promotional scheduling — run
volume-driving promotions on weekends and value-driving promotions midweek.

---

## 🎯 Recommendations

1. **Double down on Electronics & Home** — Allocate 40%+ of marketing budget
2. **Target 26-35 demographic** — Personalized campaigns for highest-AOV segment
3. **Improve shipping logistics** — Target < 7 days average for rating improvement
4. **Promote digital payments** — Higher AOV customers prefer digital methods
5. **Optimize promotion timing** — Volume promotions on weekends, value promotions midweek

---

## 📊 Methodology Notes

- **Outlier treatment**: IQR method with 1.5x multiplier; outliers were capped (winsorized)
  rather than removed to preserve data volume
- **Missing value strategy**: Median imputation for numeric columns; mode for categorical;
  rows with missing dates were dropped (critical column)
- **Validation**: All price values validated as positive; ages constrained to 13-100;
  ratings to 1-5 scale
- **Feature engineering**: Added total_amount, final_amount (post-discount), age_group,
  order_month, quarter, day_of_week, and weekend flag

---

*Report generated as part of the E-Commerce Sales Analytics project.*
*Data Cleaning & Visualization | Python, Pandas, Matplotlib, Seaborn, Streamlit*
