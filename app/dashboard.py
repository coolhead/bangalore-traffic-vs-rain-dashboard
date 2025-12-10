"""
Bengaluru Traffic vs Rainfall Dashboard

An interactive Streamlit dashboard analyzing the relationship between rainfall
and traffic congestion in Bengaluru using synthetic data modeled after real patterns.
"""

import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, time

# Configure page
st.set_page_config(
    page_title="Bengaluru Traffic vs Rainfall",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "bangalore_traffic_rain_sample.csv"
PEAK_MORNING = (8, 10)
PEAK_EVENING = (17, 20)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(path: Path) -> pd.DataFrame:
    """
    Load and preprocess the traffic and rainfall data.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame with additional time-based columns
    """
    try:
        df = pd.read_csv(path)
        
        # Convert datetime and create time-based features
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
        df["month"] = df["datetime"].dt.month_name().str[:3]
        df["dow"] = df["datetime"].dt.day_name().str[:3]
        
        # Create categorical features for better analysis
        df["rain_category"] = pd.cut(
            df["rain_mm"], 
            bins=[-0.1, 0, 1, 3, float('inf')], 
            labels=["No Rain", "Light", "Moderate", "Heavy"]
        )
        
        df["traffic_category"] = pd.cut(
            df["traffic_index"], 
            bins=[0, 0.3, 0.6, 0.8, 1.0], 
            labels=["Low", "Medium", "High", "Severe"]
        )
        
        # Add peak hour indicator
        df["is_peak"] = (
            ((df["hour"] >= PEAK_MORNING[0]) & (df["hour"] <= PEAK_MORNING[1])) |
            ((df["hour"] >= PEAK_EVENING[0]) & (df["hour"] <= PEAK_EVENING[1]))
        )
        
        return df
        
    except FileNotFoundError:
        st.error(f"Data file not found at {path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()


@st.cache_data
def filter_data(
    df: pd.DataFrame,
    areas: List[str],
    months: List[str],
    days: List[str],
    peak_only: bool,
    date_range: Optional[Tuple[datetime, datetime]] = None
) -> pd.DataFrame:
    """
    Apply filters to the dataset with caching for performance.
    
    Args:
        df: Source DataFrame
        areas: Selected areas
        months: Selected months
        days: Selected days of week
        peak_only: Whether to filter for peak hours only
        date_range: Optional date range filter
        
    Returns:
        Filtered DataFrame
    """
    filtered = df[
        df["area"].isin(areas) &
        df["month"].isin(months) &
        df["dow"].isin(days)
    ].copy()
    
    if peak_only:
        filtered = filtered[filtered["is_peak"]]
    
    if date_range:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["datetime"].dt.date >= start_date.date()) &
            (filtered["datetime"].dt.date <= end_date.date())
        ]
    
    return filtered


@st.cache_data
def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate key metrics from the filtered data.
    
    Args:
        df: Filtered DataFrame
        
    Returns:
        Dictionary containing calculated metrics
    """
    if len(df) < 2:
        return {"correlation": None, "avg_traffic": None, "avg_rain": None}
    
    correlation = df["rain_mm"].corr(df["traffic_index"])
    avg_traffic = df["traffic_index"].mean()
    avg_rain = df["rain_mm"].mean()
    max_traffic = df["traffic_index"].max()
    
    # Calculate rain impact
    no_rain_traffic = df[df["rain_mm"] == 0]["traffic_index"].mean()
    rain_traffic = df[df["rain_mm"] > 0]["traffic_index"].mean()
    rain_impact = ((rain_traffic - no_rain_traffic) / no_rain_traffic * 100) if no_rain_traffic > 0 else 0
    
    return {
        "correlation": correlation,
        "avg_traffic": avg_traffic,
        "avg_rain": avg_rain,
        "max_traffic": max_traffic,
        "rain_impact": rain_impact,
        "total_records": len(df)
    }


def create_time_series_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Create an enhanced time series chart showing rainfall and traffic patterns.
    
    Args:
        df: Filtered DataFrame
        
    Returns:
        Altair chart object
    """
    base = alt.Chart(df).add_selection(
        alt.selection_interval(bind='scales')
    )
    
    # Traffic line chart
    traffic_line = base.mark_line(
        point=alt.OverlayMarkDef(size=60, filled=True),
        strokeWidth=3
    ).encode(
        x=alt.X("datetime:T", title="Date & Time", axis=alt.Axis(format="%b %d, %H:%M")),
        y=alt.Y("traffic_index:Q", title="Traffic Index", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("area:N", title="Area", scale=alt.Scale(scheme="category10")),
        tooltip=[
            alt.Tooltip("datetime:T", format="%B %d, %Y %H:%M"),
            "area:N",
            alt.Tooltip("traffic_index:Q", format=".2f"),
            alt.Tooltip("rain_mm:Q", format=".1f"),
            "dow:N"
        ]
    )
    
    # Rainfall bars
    rain_bars = base.mark_bar(
        opacity=0.6,
        width={"band": 0.8}
    ).encode(
        x="datetime:T",
        y=alt.Y("rain_mm:Q", title="Rainfall (mm)", scale=alt.Scale(domain=[0, df["rain_mm"].max() * 1.1])),
        color=alt.value("#3498db"),
        tooltip=["datetime:T", alt.Tooltip("rain_mm:Q", format=".1f")]
    )
    
    # Combine charts with independent scales
    chart = alt.layer(rain_bars, traffic_line).resolve_scale(
        y="independent"
    ).properties(
        width="container",
        height=400,
        title=alt.TitleParams(
            text="Traffic Index and Rainfall Over Time",
            fontSize=16,
            anchor="start"
        )
    )
    
    return chart


def create_scatter_plot(df: pd.DataFrame) -> alt.Chart:
    """
    Create an enhanced scatter plot showing rainfall vs traffic relationship.
    
    Args:
        df: Filtered DataFrame
        
    Returns:
        Altair chart object
    """
    # Base scatter plot
    scatter = alt.Chart(df).mark_circle(
        size=100,
        opacity=0.7,
        stroke="white",
        strokeWidth=1
    ).encode(
        x=alt.X("rain_mm:Q", title="Rainfall (mm)", scale=alt.Scale(nice=True)),
        y=alt.Y("traffic_index:Q", title="Traffic Index", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "area:N", 
            title="Area",
            scale=alt.Scale(scheme="category10")
        ),
        size=alt.Size(
            "hour:Q",
            title="Hour of Day",
            scale=alt.Scale(range=[50, 200])
        ),
        tooltip=[
            alt.Tooltip("datetime:T", format="%B %d, %Y %H:%M"),
            "area:N",
            alt.Tooltip("rain_mm:Q", format=".1f"),
            alt.Tooltip("traffic_index:Q", format=".2f"),
            "dow:N",
            "hour:Q"
        ]
    ).add_selection(
        alt.selection_interval()
    )
    
    # Add trend line
    trend_line = alt.Chart(df).mark_line(
        color="red",
        strokeDash=[5, 5],
        strokeWidth=2
    ).transform_regression(
        "rain_mm", "traffic_index"
    ).encode(
        x="rain_mm:Q",
        y="traffic_index:Q"
    )
    
    chart = (scatter + trend_line).properties(
        width="container",
        height=400,
        title=alt.TitleParams(
            text="Traffic Index vs Rainfall (with trend line)",
            fontSize=16,
            anchor="start"
        )
    ).interactive()
    
    return chart


def create_heatmap(df: pd.DataFrame) -> alt.Chart:
    """
    Create a heatmap showing traffic patterns by hour and day of week.
    
    Args:
        df: Filtered DataFrame
        
    Returns:
        Altair chart object
    """
    # Aggregate data for heatmap
    heatmap_data = df.groupby(["dow", "hour"])["traffic_index"].mean().reset_index()
    
    # Define day order
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X("hour:O", title="Hour of Day"),
        y=alt.Y("dow:O", title="Day of Week", sort=day_order),
        color=alt.Color(
            "traffic_index:Q",
            title="Avg Traffic Index",
            scale=alt.Scale(scheme="reds", domain=[0, 1])
        ),
        tooltip=[
            "dow:O",
            "hour:O",
            alt.Tooltip("traffic_index:Q", format=".2f")
        ]
    ).properties(
        width="container",
        height=200,
        title="Average Traffic Index by Hour and Day of Week"
    )
    
    return heatmap


def main():
    """Main dashboard function."""
    
    # Header
    st.title("🚗🌧️ Bengaluru Traffic vs Rainfall Dashboard")
    st.markdown("""
    This interactive dashboard analyzes the relationship between **rainfall** and **traffic congestion** 
    in Bengaluru. The data is synthetic but modeled after real-world traffic patterns from key areas 
    like Silk Board, Outer Ring Road, and Whitefield.
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data(DATA_PATH)
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Area selection
    areas = sorted(df["area"].unique())
    selected_areas = st.sidebar.multiselect(
        "📍 Select Areas", 
        areas, 
        default=areas,
        help="Choose one or more areas to analyze"
    )
    
    # Time filters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        months = sorted(df["month"].unique())
        selected_months = st.sidebar.multiselect(
            "📅 Months", 
            months, 
            default=months
        )
    
    with col2:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        available_days = sorted(df["dow"].unique())
        selected_days = st.sidebar.multiselect(
            "📆 Days of Week", 
            available_days, 
            default=available_days
        )
    
    # Peak hours filter
    peak_only = st.sidebar.checkbox(
        "🕐 Peak Hours Only", 
        value=False,
        help="Filter for morning (8-10 AM) and evening (5-8 PM) peak hours"
    )
    
    # Date range filter
    if len(df) > 0:
        min_date = df["datetime"].min().date()
        max_date = df["datetime"].max().date()
        
        date_range = st.sidebar.date_input(
            "📊 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            date_range = (
                datetime.combine(date_range[0], time.min),
                datetime.combine(date_range[1], time.max)
            )
        else:
            date_range = None
    else:
        date_range = None
    
    # Apply filters
    if not selected_areas:
        st.warning("Please select at least one area.")
        return
    
    filtered_df = filter_data(df, selected_areas, selected_months, selected_days, peak_only, date_range)
    
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(filtered_df)
    
    # Display key metrics
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if metrics["correlation"] is not None:
            st.metric(
                "Rain-Traffic Correlation",
                f"{metrics['correlation']:.3f}",
                help="Pearson correlation between rainfall and traffic index"
            )
        else:
            st.metric("Rain-Traffic Correlation", "N/A")
    
    with col2:
        st.metric(
            "Average Traffic Index",
            f"{metrics['avg_traffic']:.2f}" if metrics['avg_traffic'] else "N/A",
            help="Mean traffic congestion level (0-1 scale)"
        )
    
    with col3:
        st.metric(
            "Average Rainfall",
            f"{metrics['avg_rain']:.1f} mm" if metrics['avg_rain'] else "N/A",
            help="Mean rainfall per hour"
        )
    
    with col4:
        st.metric(
            "Total Records",
            f"{metrics['total_records']:,}",
            help="Number of data points in current selection"
        )
    
    # Rain impact insight
    if metrics["rain_impact"] is not None and not np.isnan(metrics["rain_impact"]):
        if metrics["rain_impact"] > 5:
            st.info(f"🌧️ **Rain Impact**: Traffic increases by {metrics['rain_impact']:.1f}% during rainy periods")
        elif metrics["rain_impact"] < -5:
            st.info(f"☀️ **Weather Effect**: Traffic decreases by {abs(metrics['rain_impact']):.1f}% during rainy periods")
    
    # Charts section
    st.subheader("📈 Traffic and Rainfall Analysis")
    
    # Time series chart
    with st.container():
        st.altair_chart(create_time_series_chart(filtered_df), use_container_width=True)
    
    # Two-column layout for scatter plot and heatmap
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.altair_chart(create_scatter_plot(filtered_df), use_container_width=True)
    
    with col2:
        if len(filtered_df) > 10:  # Only show heatmap if sufficient data
            st.altair_chart(create_heatmap(filtered_df), use_container_width=True)
        else:
            st.info("Insufficient data for heatmap visualization")
    
    # Data preview
    with st.expander("🔍 View Filtered Data", expanded=False):
        st.dataframe(
            filtered_df.sort_values("datetime", ascending=False).head(100),
            use_container_width=True,
            column_config={
                "datetime": st.column_config.DatetimeColumn("Date & Time"),
                "rain_mm": st.column_config.NumberColumn("Rainfall (mm)", format="%.1f"),
                "traffic_index": st.column_config.NumberColumn("Traffic Index", format="%.2f"),
                "area": "Area",
                "rain_category": "Rain Level",
                "traffic_category": "Traffic Level"
            }
        )
    
    # Insights and observations
    st.subheader("🔍 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        **Traffic Patterns:**
        - Peak hours (8-10 AM, 5-8 PM) show highest congestion
        - Weekend traffic patterns differ from weekdays
        - Area-specific variations in traffic intensity
        """)
    
    with insights_col2:
        st.markdown("""
        **Rainfall Impact:**
        - Light rain (0-1mm) has minimal traffic impact
        - Moderate rain (1-3mm) increases congestion moderately  
        - Heavy rain (>3mm) significantly spikes traffic delays
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip**: Use the filters in the sidebar to explore specific time periods, areas, or weather conditions. "
        "Click and drag on charts to zoom in on specific data ranges."
    )


if __name__ == "__main__":
    main()
