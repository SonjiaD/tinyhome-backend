import streamlit as st
import geopandas as gpd
import numpy as np
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

# -----------------------------
# 💴 ChatGPT-style UI
# -----------------------------
st.markdown("""
    <style>
    .block-container {
        padding: 2rem 3rem;
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3, h4 {
        font-weight: 600;
        color: #202124;
        letter-spacing: -0.5px;
    }

    .stButton > button {
        background-color: #3F3F46;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 500;
        border: none;
    }

    .stSlider > div, .stNumberInput > div {
        border-radius: 10px !important;
    }

    .element-container:has(> .stDataFrame), 
    .element-container:has(> .stPlotlyChart),
    .element-container:has(> .stAltairChart),
    .element-container:has(> .stImage) {
        margin-top: 2rem;
        padding: 1rem;
        background: #f7f7f8;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# ⚡️ Load Data
# -----------------------------
@st.cache_data
def load_candidates():
    return gpd.read_file("candidates_with_features.geojson")

@st.cache_data
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min()) if series.max() != series.min() else 0

candidates = load_candidates()
score_cols = [
    "homeless_service_dist", "transit_dist", "assisted_housing_dist",
    "public_housing_dist", "city_facility_dist", "general_plan_dist",
    "water_fountain_dist", "man_water_dist", "mobile_vending_dist",
    "water_infrastructure_dist"
]

# -----------------------------
# 🎟️ State
# -----------------------------
if "weights" not in st.session_state:
    st.session_state.weights = {col: 0.1 for col in score_cols}
if "update" not in st.session_state:
    st.session_state.update = False

# -----------------------------
# 🔄 Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Map", "Data", "About"])

# -----------------------------
# 🗌️ Tab 1: Map
# -----------------------------
with tab1:
    st.title("Tiny Home Site Selector")

    st.sidebar.title("Adjust Weights of Features")
    total_weight = sum(st.session_state.weights.values())
    st.sidebar.markdown(f"**Total Weight Sum: `{total_weight:.2f}`**")
    if abs(total_weight - 1.0) > 0.01:
        st.sidebar.error("Weights must sum to 1")

    for col in score_cols:
        label = col.replace("_dist", "").replace("_", " ").title()
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.session_state.weights[col] = st.slider(
                label, 0.0, 1.0,
                value=st.session_state.weights[col],
                step=0.01,
                key=f"s_{col}",
                label_visibility="visible"
            )

        with col2:
            st.session_state.weights[col] = st.number_input(
                label, 0.0, 1.0,
                value=st.session_state.weights[col],
                step=0.01,
                key=f"n_{col}",
                label_visibility="collapsed"
            )


    if st.sidebar.button("Create Map"):
        st.session_state.update = True

    if not st.session_state.update:
        st.info("Adjust weights and click 'Create Map' to generate site rankings.")
    elif abs(total_weight - 1.0) > 0.01:
        st.error("Weights must sum to 1.")
    else:
        weights = st.session_state.weights
        weight_array = np.array(list(weights.values()))
        weight_array /= weight_array.sum()

        c = candidates.copy()
        norm_scores = []
        for col in score_cols:
            score_col = 1 / (1 + c[col])
            norm_col = min_max_normalize(score_col)
            norm_scores.append(norm_col)

        c["final_score"] = sum(w * s for w, s in zip(weight_array, norm_scores))
        ranked = c.sort_values("final_score", ascending=False).reset_index(drop=True)
        top_lots = ranked.head(500).copy()
        top_lots["rank"] = top_lots.index + 1

        # Map (with corrected projection)
        center_geom = top_lots.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
        center = [center_geom.y.mean(), center_geom.x.mean()]

        m = folium.Map(location=center, zoom_start=13)
        cluster = MarkerCluster().add_to(m)

        for _, row in top_lots.iterrows():
            if row.geometry.geom_type == "Point":
                popup = folium.Popup(
                    f"<b>ID:</b> {row.get('id', 'N/A')}<br>"
                    f"<b>Rank:</b> {row['rank']}<br>"
                    f"<b>Score:</b> {row['final_score']:.4f}",
                    max_width=300,
                )
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    color="blue",
                    fill=True,
                    fill_opacity=0.7,
                    popup=popup,
                    tooltip=f"Rank {row['rank']}"
                ).add_to(cluster)

        st_folium(m, use_container_width=True, height=750)

        # Histogram
        st.markdown("### Score Distribution (Top 500)")
        fig, ax = plt.subplots()
        ax.hist(top_lots["final_score"], bins=30, color="skyblue", edgecolor="black")
        ax.set_xlabel("Final Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

# -----------------------------
# 📊 Tab 2: Data
# -----------------------------
with tab2:
    if st.session_state.update and abs(total_weight - 1.0) < 0.01:
        st.markdown("### Top 5 Ranked Sites")
        st.dataframe(top_lots[["id", "rank", "final_score"] + score_cols].head())

        st.markdown("### Download Full Results")
        st.download_button(
            label="Download Top 500 Ranked Lots as CSV",
            data=top_lots.to_csv(index=False),
            file_name="top_500_ranked_sites.csv",
            mime="text/csv"
        )
    else:
        st.info("Run the map first to see data.")

# -----------------------------
# 📄 Tab 3: About
# -----------------------------
with tab3:
    st.title("About This Project")
    st.markdown("""
    This tool was created as part of a research initiative to identify suitable locations 
    for tiny home communities in Oakland, California. It uses geospatial data and multi-criteria 
    decision-making to rank candidate sites based on proximity to critical services like transit, 
    public housing, and city facilities.

    Built with Streamlit, Folium, and Python.
    """)
