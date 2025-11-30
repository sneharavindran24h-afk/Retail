# app.py
"""
DMart AI Dashboard (Streamlit)
- Synthetic data generator for demo
- Panels:
  - Forecast accuracy chart (matplotlib)
  - Stockout Risk Indicator
  - Top 10 Fast-Moving SKUs table
  - Inventory Health gauge (matplotlib)
  - Reorder Recommendations table
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
from datetime import datetime, timedelta

st.set_page_config(page_title="DMart AI Dashboard", layout="wide")

# --------------------------
# Helper: Synthetic Data
# --------------------------
@st.cache_data
def generate_synthetic_data(num_stores=5, num_skus=400, days=90, seed=42):
    np.random.seed(seed)
    # Create SKUs
    skus = [f"SKU_{i:04d}" for i in range(1, num_skus+1)]
    categories = np.random.choice(
        ["Staples", "Dairy", "PersonalCare", "Snacks", "Beverages", "HomeCare"],
        size=num_skus,
        p=[0.2, 0.15, 0.2, 0.2, 0.15, 0.1]
    )

    sku_master = pd.DataFrame({
        "sku": skus,
        "category": categories,
        "price": np.round(np.random.uniform(20, 500, size=num_skus), 2)
    })

    # Store profiles
    stores = [f"Store_{i}" for i in range(1, num_stores+1)]
    store_profiles = pd.DataFrame({
        "store": stores,
        "type": np.random.choice(["Urban", "SemiUrban", "Suburban"], size=num_stores, p=[0.5, 0.3, 0.2]),
        "daily_footfall": np.random.randint(500, 5000, size=num_stores)
    })

    # Time range
    end_date = datetime.today().date()
    dates = [end_date - timedelta(days=x) for x in range(days)]
    dates = sorted(dates)

    # Generate sales per sku-store-day with seasonality + noise
    records = []
    for store in stores:
        store_factor = store_profiles.loc[store_profiles.store == store, "daily_footfall"].values[0] / 2000
        for sku_idx, row in sku_master.iterrows():
            base_velocity = np.random.poisson(lam=2 + (500 / (row.price + 50)))  # cheaper items sell more
            # random SKU seasonality (0 = steady, 1 = seasonal)
            seasonality = np.random.choice([0, 1], p=[0.85, 0.15])
            for d in dates:
                day_of_week = d.weekday()
                # small weekly pattern + seasonality spikes on some random days
                weekly = 1 + 0.1*math.sin(2*math.pi*(day_of_week)/7)
                seasonal_spike = 1.0
                # introduce a festival spike randomly
                if seasonality == 1 and np.random.rand() < 0.02:
                    seasonal_spike = np.random.uniform(1.5, 2.5)
                sold = np.random.poisson(lam=max(0.1, base_velocity*weekly*store_factor*seasonal_spike))
                records.append((store, row.sku, row.category, d, sold))
    sales = pd.DataFrame(records, columns=["store", "sku", "category", "date", "units_sold"])

    # Aggregate recent stock and lead time
    sku_stock = []
    for store in stores:
        for sku in skus:
            current = np.random.randint(0, 400)
            lead = np.random.choice([1,2,3,4,7], p=[0.3,0.25,0.2,0.15,0.1])
            sku_stock.append((store, sku, current, lead))
    stock_df = pd.DataFrame(sku_stock, columns=["store", "sku", "on_hand", "lead_time"])

    return sku_master, store_profiles, sales, stock_df

sku_master, store_profiles, sales_df, stock_df = generate_synthetic_data(num_stores=8, num_skus=600, days=120)

# --------------------------
# Sidebar: Controls
# --------------------------
st.sidebar.title("DMart AI Dashboard Controls")
selected_store = st.sidebar.selectbox("Select Store", store_profiles.store.tolist())
horizon_weeks = st.sidebar.slider("Forecast horizon (weeks)", min_value=1, max_value=4, value=2)
top_n = st.sidebar.slider("Top N fast-moving SKUs", min_value=5, max_value=20, value=10)

# --------------------------
# Compute simple forecasts (naive + rolling) for demo
# --------------------------
def compute_weekly_aggregates(sales_df, store, weeks=12):
    store_sales = sales_df[sales_df.store == store].copy()
    store_sales['week'] = pd.to_datetime(store_sales.date).dt.to_period('W').apply(lambda r: r.start_time.date())
    weekly = store_sales.groupby(['sku','category','week'])['units_sold'].sum().reset_index()
    # last N weeks pivot
    return weekly

weekly = compute_weekly_aggregates(sales_df, selected_store)
recent_week = weekly['week'].max()
# Create demand series per SKU (last 8 weeks)
last_weeks = weekly[weekly.week >= (recent_week - pd.Timedelta(weeks=12))]
pivot = last_weeks.pivot_table(index='sku', columns='week', values='units_sold', fill_value=0)
pivot['historical_mean'] = pivot.mean(axis=1)
pivot['recent_mean_4w'] = pivot[pivot.columns[-4:]].mean(axis=1)
pivot = pivot.reset_index()

# For demo: Forecast = recent_mean_4w * horizon_weeks
pivot['forecast_units'] = (pivot['recent_mean_4w'] * horizon_weeks).round().astype(int)

# Merge on-hand
store_stock = stock_df[stock_df.store == selected_store]
pivot = pivot.merge(store_stock[['sku','on_hand','lead_time']], on='sku', how='left').fillna(0)

# Fast-moving SKUs by recent 4-week avg
top_fast = pivot.sort_values('recent_mean_4w', ascending=False).head(top_n)
top_fast = top_fast.merge(sku_master[['sku','category','price']], on='sku', how='left')

# Stockout risk: if forecast > on_hand => risk
pivot['stockout_risk_units'] = pivot['forecast_units'] - pivot['on_hand']
pivot['risk_level'] = pd.cut(pivot['stockout_risk_units'],
                             bins=[-999, 0, 50, 999999],
                             labels=['Green','Yellow','Red'])

# Reorder recommendation (simple): order = max(0, forecast - on_hand + safety_buffer)
pivot['safety_buffer'] = (pivot['forecast_units'] * 0.25).round().astype(int)
pivot['recommended_order'] = (pivot['forecast_units'] - pivot['on_hand'] + pivot['safety_buffer']).apply(lambda x: max(0,int(x)))
reorder_table = pivot[['sku','forecast_units','on_hand','recommended_order','risk_level','safety_buffer']].sort_values('recommended_order', ascending=False).head(30)

# Inventory health metric (0-100)
def inventory_health(pivot_df):
    # simple heuristic: proportion of top-50 SKUs that are green
    top50 = pivot_df.sort_values('recent_mean_4w', ascending=False).head(50)
    green_pct = (top50['risk_level']=='Green').sum() / max(1, len(top50))
    health = 50 + green_pct*50  # scale to 50-100
    return int(health*100/100)

health_score = inventory_health(pivot)

# --------------------------
# Layout: Top Row
# --------------------------
col1, col2, col3 = st.columns([3,2,2])

with col1:
    st.header(f"DMart Inventory Insights — {selected_store}")
    st.subheader(f"Forecast horizon: {horizon_weeks} week(s)")
    st.write("Summary: Top alerts and quick actions for store managers.")

with col2:
    st.metric("Inventory Health", f"{health_score}/100", delta="+"+str(int(health_score-75)) if health_score>75 else "")

with col3:
    # Stockout risk counts
    red_count = (pivot['risk_level']=='Red').sum()
    yellow_count = (pivot['risk_level']=='Yellow').sum()
    st.markdown("### Stockout Risk")
    st.write(f"High (Red): {red_count}")
    st.write(f"Medium (Yellow): {yellow_count}")

# --------------------------
# Forecast Accuracy Chart (matplotlib)
# --------------------------
st.markdown("## Forecast vs Actual (Sample SKUs)")
# pick top 5 SKUs for chart
sample_skus = top_fast.sku.head(5).tolist()
fig, ax = plt.subplots(figsize=(8,3.5))
weeks_plot = sorted(last_weeks['week'].unique())
x = range(len(weeks_plot))
for sku in sample_skus:
    hist = last_weeks[last_weeks.sku==sku].set_index('week').reindex(weeks_plot, fill_value=0)['units_sold'].values
    # naive forecast (historical_mean) and demo forecast
    naive = np.ones_like(hist)*hist.mean()
    demo_forecast = np.concatenate([hist[:-horizon_weeks], np.ones(horizon_weeks)*hist[-4:].mean()])[:len(hist)]
    ax.plot(x, hist, marker='o', linewidth=1, label=f"{sku} actual")
    ax.plot(x, demo_forecast, linestyle='--', linewidth=1, label=f"{sku} forecast")
ax.set_xticks(x)
ax.set_xticklabels([d.strftime("%d %b") for d in weeks_plot], rotation=45, fontsize=8)
ax.set_xlabel("Week")
ax.set_ylabel("Units sold")
ax.legend(fontsize=8, ncol=2)
st.pyplot(fig)

# --------------------------
# Top N Fast Moving SKUs
# --------------------------
st.markdown("## Top Fast-Moving SKUs")
st.dataframe(top_fast[['sku','category','price','recent_mean_4w','forecast_units']].rename(columns={
    'recent_mean_4w':'4-week avg'
}).reset_index(drop=True), use_container_width=True)

# --------------------------
# Reorder Recommendations
# --------------------------
st.markdown("## Reorder Recommendations (Top 30)")
st.dataframe(reorder_table.reset_index(drop=True), use_container_width=True)

# --------------------------
# Inventory Health Gauge (matplotlib)
# --------------------------
st.markdown("## Inventory Health Gauge")
fig2, ax2 = plt.subplots(figsize=(3.5,2.5))
# Draw a semicircle gauge
theta = np.linspace(-math.pi/1, 0, 100)
r = 1
ax2.plot(np.cos(theta), np.sin(theta), linewidth=10)
# Needle
angle = -math.pi + (math.pi * (health_score/100))
nx, ny = math.cos(angle), math.sin(angle)
ax2.plot([0, nx], [0, ny], linewidth=3)
ax2.text(0, -0.2, f"Health: {health_score}/100", ha='center', va='center')
ax2.axis('off')
st.pyplot(fig2)

# --------------------------
# Sidebar: Download CSVs
# --------------------------
st.sidebar.markdown("### Export")
if st.sidebar.button("Download Reorder CSV"):
    csv = reorder_table.to_csv(index=False)
    st.sidebar.download_button("Download reorder.csv", csv, "reorder.csv", "text/csv")

st.sidebar.markdown("### Notes")
st.sidebar.write("This is a demo dashboard using synthetic data. In a production setup, replace the data loader with actual POS and stock data, and integrate model outputs (XGBoost) for forecasts.")

