
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Parking Analytics",
    page_icon="🅿️",
    layout="wide"
)

st.title("🅿️ Smart Parking Analytics Dashboard")
st.subheader("Parking Usage, Occupancy and Operational Analysis")

st.markdown(
    "An interactive dashboard for analyzing parking occupancy, "
    "vehicle usage, revenue and parking operations."
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    file_path = (
        Path(__file__).parent /
        "smart_parking_usage_occupancy_analytics_demo_visualization.csv"
    )

    df = pd.read_csv(file_path)

    df["date_time"] = pd.to_datetime(
        df["date_time"], errors="coerce"
    )

    numeric_columns = [
        "total_slots",
        "occupied_slots",
        "parking_fee_collected",
        "avg_parking_duration_minutes",
        "entry_count",
        "exit_count"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df["occupied_slots_original"] = df["occupied_slots"]

    df["capacity_error"] = (
        df["occupied_slots"] > df["total_slots"]
    )

    df["occupied_slots"] = np.minimum(
        df["occupied_slots"],
        df["total_slots"]
    )

    df["occupancy_rate_percent"] = (
        df["occupied_slots"] /
        df["total_slots"].replace(0, np.nan) * 100
    ).round(2)

    df["hour"] = df["date_time"].dt.hour
    df["date"] = df["date_time"].dt.date

    return df


df = load_data()

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Dashboard Filters")

zones = ["All"] + sorted(
    df["zone"].dropna().unique().tolist()
)

selected_zone = st.sidebar.selectbox(
    "Select Parking Zone",
    zones
)

vehicle_types = ["All"] + sorted(
    df["vehicle_type"].dropna().unique().tolist()
)

selected_vehicle = st.sidebar.selectbox(
    "Select Vehicle Type",
    vehicle_types
)

filtered_df = df.copy()

if selected_zone != "All":
    filtered_df = filtered_df[
        filtered_df["zone"] == selected_zone
    ]

if selected_vehicle != "All":
    filtered_df = filtered_df[
        filtered_df["vehicle_type"] == selected_vehicle
    ]

if filtered_df.empty:
    st.warning("No data available for these filters.")
    st.stop()

# ==========================================
# KPI CALCULATIONS
# ==========================================

average_occupancy = filtered_df[
    "occupancy_rate_percent"
].mean()

total_revenue = filtered_df[
    "parking_fee_collected"
].sum()

total_violations = (
    filtered_df["violation_reported"] == "Yes"
).sum()

average_duration = filtered_df[
    "avg_parking_duration_minutes"
].mean()

# ==========================================
# KPI CARDS
# ==========================================

st.markdown("### 📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Occupancy",
    f"{average_occupancy:.2f}%"
)

col2.metric(
    "Total Revenue",
    f"₹{total_revenue:,.0f}"
)

col3.metric(
    "Total Violations",
    f"{total_violations:,}"
)

col4.metric(
    "Average Duration",
    f"{average_duration:.1f} min"
)

st.divider()

# ==========================================
# PLOT 1 - OCCUPANCY BY ZONE
# BAR CHART
# ==========================================

st.markdown("### 1. Occupancy by Zone")

zone_analysis = (
    filtered_df.groupby("zone")["occupancy_rate_percent"]
    .mean().sort_values(ascending=False)
)

fig1, ax1 = plt.subplots(figsize=(8, 3))

bars = ax1.bar(
    zone_analysis.index,
    zone_analysis.values,
    color="#2563EB",
    width=0.55
)

ax1.bar_label(
    bars,
    fmt="%.1f%%",
    padding=3
)

ax1.set_title("Average Parking Occupancy by Zone")
ax1.set_xlabel("Parking Zone")
ax1.set_ylabel("Occupancy (%)")
ax1.set_ylim(0, 110)
ax1.tick_params(axis="x", rotation=20)
ax1.grid(axis="y", alpha=0.25)

plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

st.divider()

# ==========================================
# PLOT 2 - OCCUPANCY BY HOUR
# LINE CHART
# ==========================================

st.markdown("### 2. Hourly Occupancy Trend")

hour_analysis = (
    filtered_df.groupby("hour")["occupancy_rate_percent"]
    .mean().sort_index()
)

fig2, ax2 = plt.subplots(figsize=(8, 3))

ax2.plot(
    hour_analysis.index,
    hour_analysis.values,
    marker="o",
    linewidth=2,
    color="#0D9488"
)

ax2.set_title("Average Parking Occupancy by Hour")
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Occupancy (%)")
ax2.set_xticks(range(0, 24, 2))
ax2.grid(alpha=0.25)

plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.divider()

# ==========================================
# PLOT 3 - DAILY OCCUPANCY
# LINE CHART
# ==========================================

st.markdown("### 3. Daily Occupancy Trend")

daily_analysis = (
    filtered_df.groupby("date")["occupancy_rate_percent"]
    .mean().sort_index()
)

fig3, ax3 = plt.subplots(figsize=(8, 3))

ax3.plot(
    daily_analysis.index,
    daily_analysis.values,
    color="#7C3AED",
    linewidth=1.8,
    marker="."
)

rolling_average = daily_analysis.rolling(
    window=7,
    min_periods=1
).mean()

ax3.plot(
    daily_analysis.index,
    rolling_average,
    color="#F97316",
    linewidth=2,
    label="7-day average"
)

ax3.set_title("Daily Parking Occupancy")
ax3.set_xlabel("Date")
ax3.set_ylabel("Occupancy (%)")
ax3.tick_params(axis="x", rotation=35)
ax3.legend()
ax3.grid(alpha=0.25)

plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

st.divider()

# ==========================================
# PLOT 4 - VEHICLE OCCUPANCY
# BOX PLOT
# ==========================================

st.markdown("### 4. Vehicle Occupancy Distribution")

vehicle_types_data = sorted(
    filtered_df["vehicle_type"].dropna().unique()
)

vehicle_data = [
    filtered_df.loc[
        filtered_df["vehicle_type"] == vehicle,
        "occupancy_rate_percent"
    ].dropna()
    for vehicle in vehicle_types_data
]

fig4, ax4 = plt.subplots(figsize=(8, 3))

ax4.boxplot(
    vehicle_data,
    patch_artist=True,
    showmeans=True
)

ax4.set_xticks(
    range(1, len(vehicle_types_data) + 1)
)

ax4.set_xticklabels(vehicle_types_data)

ax4.set_title("Occupancy Distribution by Vehicle Type")
ax4.set_xlabel("Vehicle Type")
ax4.set_ylabel("Occupancy (%)")
ax4.grid(axis="y", alpha=0.25)

plt.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

st.divider()

# ==========================================
# PLOT 5 - VIOLATION DISTRIBUTION
# PIE CHART
# ==========================================

st.markdown("### 5. Parking Violation Distribution")

violation_analysis = (
    filtered_df["violation_reported"]
    .fillna("Unknown")
    .value_counts()
)

fig5, ax5 = plt.subplots(figsize=(6, 3.5))

ax5.pie(
    violation_analysis.values,
    labels=violation_analysis.index,
    autopct="%1.1f%%",
    startangle=90
)

ax5.set_title("Violation vs Non-Violation Records")

plt.tight_layout()
st.pyplot(fig5)
plt.close(fig5)

st.divider()

# ==========================================
# PLOT 6 - REVENUE BY ZONE
# HORIZONTAL BAR CHART
# ==========================================

st.markdown("### 6. Parking Revenue by Zone")

revenue_analysis = (
    filtered_df.groupby("zone")["parking_fee_collected"]
    .sum().sort_values()
)

fig6, ax6 = plt.subplots(figsize=(8, 3))

bars = ax6.barh(
    revenue_analysis.index,
    revenue_analysis.values,
    color="#0D9488",
    height=0.55
)

ax6.bar_label(
    bars,
    labels=[
        f"₹{value:,.0f}"
        for value in revenue_analysis.values
    ],
    padding=3
)

ax6.set_title("Total Revenue by Parking Zone")
ax6.set_xlabel("Revenue (₹)")
ax6.grid(axis="x", alpha=0.25)

plt.tight_layout()
st.pyplot(fig6)
plt.close(fig6)

st.divider()

# ==========================================
# PLOT 7 - PARKING DURATION
# HISTOGRAM
# ==========================================

st.markdown("### 7. Parking Duration Distribution")

duration_data = filtered_df[
    "avg_parking_duration_minutes"
].dropna()

fig7, ax7 = plt.subplots(figsize=(8, 3))

ax7.hist(
    duration_data,
    bins=15,
    color="#0D9488",
    edgecolor="white"
)

ax7.set_title("Distribution of Parking Duration")
ax7.set_xlabel("Parking Duration (Minutes)")
ax7.set_ylabel("Frequency")
ax7.grid(axis="y", alpha=0.25)

plt.tight_layout()
st.pyplot(fig7)
plt.close(fig7)

st.divider()

# ==========================================
# PLOT 8 - ENTRIES VS EXITS
# SCATTER PLOT
# ==========================================

st.markdown("### 8. Parking Entries vs Exits")

entry_exit = (
    filtered_df.groupby("zone")[
        ["entry_count", "exit_count"]
    ].sum()
)

fig8, ax8 = plt.subplots(figsize=(8, 3))

ax8.scatter(
    entry_exit["entry_count"],
    entry_exit["exit_count"],
    s=90,
    color="#7C3AED",
    edgecolor="white"
)

for zone, row in entry_exit.iterrows():
    ax8.annotate(
        zone,
        (row["entry_count"], row["exit_count"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

ax8.set_title("Entries vs Exits Across Parking Zones")
ax8.set_xlabel("Total Entries")
ax8.set_ylabel("Total Exits")
ax8.grid(alpha=0.25)

plt.tight_layout()
st.pyplot(fig8)
plt.close(fig8)

st.divider()

# ==========================================
# PLOT 9 - OCCUPANCY HEATMAP
# ==========================================

st.markdown("### 9. Parking Occupancy Heatmap")

heatmap_data = filtered_df.pivot_table(
    index="zone",
    columns="hour",
    values="occupancy_rate_percent",
    aggfunc="mean"
)

fig9, ax9 = plt.subplots(figsize=(9, 3.5))

im = ax9.imshow(
    heatmap_data,
    aspect="auto",
    cmap="YlGnBu",
    interpolation="nearest",
    vmin=0,
    vmax=100
)

ax9.set_yticks(
    np.arange(len(heatmap_data.index))
)

ax9.set_yticklabels(
    heatmap_data.index
)

ax9.set_xticks(
    np.arange(len(heatmap_data.columns))
)

ax9.set_xticklabels(
    heatmap_data.columns
)

ax9.set_xlabel("Hour of Day")
ax9.set_ylabel("Parking Zone")
ax9.set_title("Occupancy by Zone and Hour")

fig9.colorbar(
    im,
    ax=ax9,
    label="Occupancy (%)"
)

plt.tight_layout()
st.pyplot(fig9)
plt.close(fig9)


# DATA TABLE


st.divider()

st.markdown("### 📋 Filtered Parking Dataset")

st.caption(
    f"Showing {len(filtered_df):,} records"
)

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)


# FOOTER


st.divider()

st.caption(
    "Smart Parking Analytics | "
    "Python | Pandas | NumPy | Matplotlib | Streamlit"
)
