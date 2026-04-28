"""
=============================================================================
E-Commerce Sales Data Generator
=============================================================================
Generates a realistic raw dataset with intentional data quality issues for
practicing data cleaning techniques.

Issues injected:
    - Missing values (NaN) in multiple columns
    - Outliers (extreme values in price and shipping)
    - Duplicate rows (exact and near-duplicates)
    - Inconsistent formatting (mixed case, whitespace, date formats)
    - Invalid data (negative prices, impossible ages, future dates)
    - Mixed data types (numbers stored as strings)
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# ── Configuration ──────────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

NUM_RECORDS = 5000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ecommerce_sales_raw.csv")


# ── Helper Data ────────────────────────────────────────────────────────────────
CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Books",
    "Sports & Outdoors", "Beauty", "Toys & Games", "Automotive"
]

# Intentionally messy category names (inconsistent formatting)
MESSY_CATEGORIES = [
    "electronics", "Electronics", "ELECTRONICS", " Electronics ",
    "Clothing", "clothing", "CLOTHING", "Clothng",  # Typo
    "Home & Kitchen", "home & kitchen", "Home &  Kitchen",
    "Books", "books", "BOOKS", "Book",
    "Sports & Outdoors", "sports & outdoors", "Sports",
    "Beauty", "beauty", "BEAUTY", " Beauty",
    "Toys & Games", "toys & games", "Toys",
    "Automotive", "automotive", "AUTOMOTIVE"
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
    "London", "Manchester", "Birmingham", "Leeds", "Glasgow"
]

PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Cash on Delivery", "Bank Transfer"]

CUSTOMER_NAMES = [
    "Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Ross",
    "Edward Norton", "Fiona Apple", "George Lucas", "Hannah Montana",
    "Ivan Drago", "Julia Roberts", "Kevin Hart", "Lisa Simpson",
    "Michael Scott", "Nancy Drew", "Oscar Wilde", "Patricia Arquette",
    "Quincy Jones", "Rachel Green", "Steve Rogers", "Tina Turner",
    "Uma Thurman", "Victor Hugo", "Wendy Williams", "Xavier Woods",
    "Yolanda Adams", "Zach Efron", "Amit Sharma", "Priya Patel",
    "Rahul Gupta", "Sneha Reddy", "James Wilson", "Emily Chen",
    "Mohammed Ali", "Sarah Connor", "David Kim", "Maria Garcia"
]


def generate_dates(n: int) -> list:
    """Generate a list of dates with various format inconsistencies."""
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    delta = (end_date - start_date).days

    dates = []
    date_formats = [
        "%Y-%m-%d",       # 2024-01-15
        "%d/%m/%Y",       # 15/01/2024
        "%m-%d-%Y",       # 01-15-2024
        "%d %b %Y",       # 15 Jan 2024
        "%B %d, %Y",      # January 15, 2024
    ]

    for _ in range(n):
        random_days = random.randint(0, delta)
        date = start_date + timedelta(days=random_days)

        # 80% standard format, 20% varied formats
        if random.random() < 0.8:
            dates.append(date.strftime("%Y-%m-%d"))
        else:
            fmt = random.choice(date_formats)
            dates.append(date.strftime(fmt))

    # Inject some future dates (invalid)
    future_indices = random.sample(range(n), int(n * 0.02))
    for idx in future_indices:
        future_date = datetime.now() + timedelta(days=random.randint(30, 365))
        dates[idx] = future_date.strftime("%Y-%m-%d")

    return dates


def generate_prices(n: int) -> list:
    """Generate prices with outliers and invalid values."""
    prices = []
    for _ in range(n):
        rand = random.random()
        if rand < 0.03:
            # Negative prices (invalid)
            prices.append(round(-random.uniform(1, 100), 2))
        elif rand < 0.06:
            # Extreme outliers
            prices.append(round(random.uniform(5000, 50000), 2))
        elif rand < 0.10:
            # Store as string (mixed types)
            price = round(random.uniform(5, 500), 2)
            prices.append(str(price))
        elif rand < 0.12:
            # Price with currency symbol (inconsistent)
            price = round(random.uniform(5, 500), 2)
            prices.append(f"₹{price}")
        else:
            # Normal prices
            prices.append(round(random.uniform(5, 500), 2))
    return prices


def generate_dataset() -> pd.DataFrame:
    """Generate the complete raw dataset with intentional quality issues."""
    print("=" * 60)
    print("  E-Commerce Sales Data Generator")
    print("=" * 60)
    print(f"\n📊 Generating {NUM_RECORDS} records with data quality issues...\n")

    data = {
        "order_id": list(range(1001, 1001 + NUM_RECORDS)),
        "order_date": generate_dates(NUM_RECORDS),
        "customer_name": [random.choice(CUSTOMER_NAMES) for _ in range(NUM_RECORDS)],
        "customer_email": [],
        "customer_age": [],
        "city": [random.choice(CITIES) for _ in range(NUM_RECORDS)],
        "category": [random.choice(MESSY_CATEGORIES) for _ in range(NUM_RECORDS)],
        "product_name": [],
        "quantity": [],
        "unit_price": generate_prices(NUM_RECORDS),
        "discount_pct": [],
        "payment_method": [random.choice(PAYMENT_METHODS) for _ in range(NUM_RECORDS)],
        "shipping_days": [],
        "rating": [],
    }

    # ── Customer emails (with some invalid) ────────────────────────────────
    for i in range(NUM_RECORDS):
        name = data["customer_name"][i].lower().replace(" ", ".")
        domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "company.com"]
        email = f"{name}@{random.choice(domains)}"
        if random.random() < 0.05:
            # Invalid emails
            email = random.choice([name, f"{name}@", "@gmail.com", "not_an_email", ""])
        data["customer_email"].append(email)

    # ── Customer ages (with invalid values) ────────────────────────────────
    for _ in range(NUM_RECORDS):
        rand = random.random()
        if rand < 0.03:
            data["customer_age"].append(random.choice([-5, 0, 150, 200, 999]))
        elif rand < 0.06:
            data["customer_age"].append(str(random.randint(18, 70)))  # String type
        else:
            data["customer_age"].append(random.randint(18, 70))

    # ── Product names ──────────────────────────────────────────────────────
    product_prefixes = {
        "Electronics": ["Wireless", "Smart", "Digital", "Pro", "Ultra"],
        "Clothing": ["Classic", "Slim-Fit", "Premium", "Casual", "Designer"],
        "Home & Kitchen": ["Deluxe", "Modern", "Compact", "Stainless", "Ceramic"],
        "Books": ["The Art of", "Introduction to", "Advanced", "Complete Guide to", "Essential"],
        "Sports & Outdoors": ["Pro Series", "Adventure", "Elite", "Performance", "Trail"],
        "Beauty": ["Luxe", "Natural", "Organic", "Premium", "Radiant"],
        "Toys & Games": ["Fun", "Creative", "Educational", "Interactive", "Classic"],
        "Automotive": ["Heavy Duty", "Universal", "Premium", "All-Weather", "Performance"],
    }

    product_suffixes = {
        "Electronics": ["Headphones", "Speaker", "Charger", "Mouse", "Keyboard", "Monitor", "Tablet"],
        "Clothing": ["T-Shirt", "Jeans", "Jacket", "Sneakers", "Dress", "Hoodie"],
        "Home & Kitchen": ["Blender", "Toaster", "Pan Set", "Knife Set", "Coffee Maker"],
        "Books": ["Python", "Data Science", "Machine Learning", "Business", "History"],
        "Sports & Outdoors": ["Running Shoes", "Yoga Mat", "Water Bottle", "Backpack", "Tennis Racket"],
        "Beauty": ["Moisturizer", "Serum", "Sunscreen", "Lip Balm", "Face Wash"],
        "Toys & Games": ["Board Game", "Puzzle", "Building Set", "Action Figure", "Card Game"],
        "Automotive": ["Floor Mats", "Phone Mount", "Dash Cam", "Air Freshener", "Seat Cover"],
    }

    for i in range(NUM_RECORDS):
        # Map messy category to standard for product name generation
        cat = data["category"][i].strip().title()
        matched = False
        for std_cat in CATEGORIES:
            if cat.lower().startswith(std_cat.lower()[:4]):
                prefix = random.choice(product_prefixes[std_cat])
                suffix = random.choice(product_suffixes[std_cat])
                data["product_name"].append(f"{prefix} {suffix}")
                matched = True
                break
        if not matched:
            data["product_name"].append(f"Generic Product {random.randint(100, 999)}")

    # ── Quantity (with some zeros and negatives) ───────────────────────────
    for _ in range(NUM_RECORDS):
        rand = random.random()
        if rand < 0.02:
            data["quantity"].append(0)
        elif rand < 0.04:
            data["quantity"].append(-random.randint(1, 5))
        elif rand < 0.06:
            data["quantity"].append(random.randint(50, 200))  # Outlier
        else:
            data["quantity"].append(random.randint(1, 10))

    # ── Discount percentage ────────────────────────────────────────────────
    for _ in range(NUM_RECORDS):
        rand = random.random()
        if rand < 0.03:
            data["discount_pct"].append(random.choice([-10, 110, 150]))  # Invalid
        elif rand < 0.60:
            data["discount_pct"].append(0)  # No discount
        else:
            data["discount_pct"].append(random.choice([5, 10, 15, 20, 25, 30, 40, 50]))

    # ── Shipping days ──────────────────────────────────────────────────────
    for _ in range(NUM_RECORDS):
        rand = random.random()
        if rand < 0.03:
            data["shipping_days"].append(random.choice([-2, 0, 60, 90, 120]))  # Invalid/outlier
        else:
            data["shipping_days"].append(random.randint(1, 14))

    # ── Rating ─────────────────────────────────────────────────────────────
    for _ in range(NUM_RECORDS):
        rand = random.random()
        if rand < 0.03:
            data["rating"].append(random.choice([0, -1, 6, 10]))  # Invalid
        else:
            data["rating"].append(round(random.uniform(1, 5), 1))

    df = pd.DataFrame(data)

    # ── Inject Missing Values ──────────────────────────────────────────────
    missing_config = {
        "customer_name": 0.02,
        "customer_email": 0.05,
        "customer_age": 0.08,
        "city": 0.03,
        "category": 0.04,
        "unit_price": 0.06,
        "discount_pct": 0.03,
        "shipping_days": 0.05,
        "rating": 0.10,
    }

    for col, pct in missing_config.items():
        mask = np.random.random(len(df)) < pct
        df.loc[mask, col] = np.nan

    # ── Inject Duplicate Rows ─────────────────────────────────────────────
    # Exact duplicates (3% of data)
    num_exact_dupes = int(NUM_RECORDS * 0.03)
    dupe_indices = random.sample(range(len(df)), num_exact_dupes)
    exact_dupes = df.iloc[dupe_indices].copy()
    df = pd.concat([df, exact_dupes], ignore_index=True)

    # Near-duplicates (slightly different values, 2% of data)
    num_near_dupes = int(NUM_RECORDS * 0.02)
    near_dupe_indices = random.sample(range(len(df)), num_near_dupes)
    near_dupes = df.iloc[near_dupe_indices].copy()
    # Modify slightly
    for idx in near_dupes.index:
        if random.random() < 0.5:
            near_dupes.at[idx, "order_id"] = near_dupes.at[idx, "order_id"]  # Same ID
            near_dupes.at[idx, "quantity"] = near_dupes.at[idx, "quantity"]
            if pd.notna(near_dupes.at[idx, "rating"]):
                try:
                    near_dupes.at[idx, "rating"] = round(float(near_dupes.at[idx, "rating"]) + 0.1, 1)
                except (ValueError, TypeError):
                    pass
    df = pd.concat([df, near_dupes], ignore_index=True)

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Print Summary ──────────────────────────────────────────────────────
    print("✅ Dataset generated successfully!\n")
    print(f"   📋 Total records: {len(df):,}")
    print(f"   📊 Columns: {len(df.columns)}")
    print(f"   ❌ Missing values: {df.isnull().sum().sum():,}")
    print(f"   🔄 Approximate duplicates injected: {num_exact_dupes + num_near_dupes}")
    print(f"\n   Issues injected:")
    print(f"     • Missing values in {len(missing_config)} columns")
    print(f"     • Inconsistent date formats")
    print(f"     • Mixed category naming (case, typos, whitespace)")
    print(f"     • Negative/zero prices and quantities")
    print(f"     • Invalid ages, ratings, discount percentages")
    print(f"     • Mixed data types (numbers as strings)")
    print(f"     • Invalid email formats")
    print(f"     • Outliers in price, quantity, and shipping")
    print(f"     • Exact and near-duplicate rows")

    return df


def save_dataset(df: pd.DataFrame) -> None:
    """Save the dataset to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n   💾 Saved to: {OUTPUT_FILE}")
    print(f"   📁 File size: {file_size:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    df = generate_dataset()
    save_dataset(df)
