# 🧹 E-Commerce Sales Analytics & Data Cleaning Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

An end-to-end, industry-grade data science project demonstrating advanced data cleaning, exploratory analysis, and interactive storytelling. This project transforms a highly messy synthetic e-commerce dataset into actionable business insights.

## 📊 Project Highlights

- **Currency**: Fully localized to **Indian Rupees (₹)**.
- **Robust Pipeline**: 7-step modular architecture to handle real-world data issues.
- **Interactive Dashboard**: Sleek Streamlit UI for dynamic slice-and-dice analytics.
- **Visual Storytelling**: High-quality visual reports tracking the before/after data impact.

---

## 🛠️ Tech Stack & Dependencies

| Category | Tools Used |
|---|---|
| **Core Engine** | `Python 3.14` |
| **Data Wrangling** | `Pandas`, `NumPy`, `SciPy` |
| **Visualization** | `Matplotlib`, `Seaborn`, `Plotly` |
| **Dashboarding** | `Streamlit` |

---

## 🏗️ The 7-Step Cleaning Pipeline

The project handles raw data vulnerabilities across the following distinct stages:

1. **Duplicate Elimination**: Detects and drops both exact matching and fuzzy/near-duplicates.
2. **Text Standardization**: Unifies mixed cases, strips whitespace, and maps messy categories.
3. **Type Safety & Conversion**: Cleans currency prefixes (`₹`, `$`), comma separators, and converts strings safely to numerical formats.
4. **Range Validations**: Enforces logical boundary rules (e.g., age limits, valid product ratings).
5. **Outlier Mitigation**: Winsorizes extreme distributions using standard IQR boundaries.
6. **Missing Values**: Employs median/mode imputation dynamically.
7. **Feature Engineering**: Derives key analytical columns like post-discount pricing.

---

## 🚀 Quick Execution Guide

### 1. Set Up Environment
```bash
# Activate your existing virtual environment (if present) or create one:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Workflow
Execute the core components sequentially:

```bash
# 1. Regenerate raw dummy e-commerce data
python src/data_generator.py

# 2. Run initial exploratory assessment
python notebooks/01_data_exploration.py

# 3. Clean the pipeline safely
python notebooks/02_data_cleaning.py

# 4. Export static summary PNG charts
python notebooks/03_visualization.py
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run notebooks/04_dashboard.py
```

## 📁 Repository Map
```
├── data/                  # Raw inputs vs. Final Cleaned results
├── notebooks/             # Step-by-step modular runtime scripts
├── src/                   # Reusable algorithms & Plot helpers
└── reports/               # Auto-generated image assets
```

## 👤 Project Maintainer
- **Subrat Tandon** — *Data Science Portfolio Initiative*
