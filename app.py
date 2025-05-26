import streamlit as st
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import pandas as pd
from urllib.parse import parse_qs
import os

# -----------------------------
#trying to make entire layout full width
#removes default width cap, makes all tabs span full width
st.set_page_config(layout = "wide")

# -----------------------------
#trying to add lightweight warm-up endpoint
# Optional warm-up support (advanced use, only for backend triggers)
query_params = st.query_params
if query_params.get("warmup", ["false"])[0].lower() == "true":
    _ = load_candidates()
    st.write("Warmed up!")
    st.stop()
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
tab1, tab2, tab3 = st.tabs(["Map", "Data", "Histogram"])

# -----------------------------
# 🗌️ Tab 1: Map
# -----------------------------
with tab1:

    st.sidebar.title("Adjust Weights of Features")

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
                label="",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.weights[col],
                step=0.01,
                key=f"n_{col}",
                label_visibility="collapsed"
            )

    total_weight = sum(st.session_state.weights.values())
    st.sidebar.markdown(f"**Total Weight Sum: `{total_weight:.2f}`**")

    if abs(total_weight - 1.0) > 0.01:
        st.sidebar.error("Weights must sum to 1")

    if st.sidebar.button("Create Map"):
        st.session_state.update = True

    if not st.session_state.update:
        st.info("Adjust weights and click 'Create Map' to generate site rankings.")
    elif abs(total_weight - 1.0) > 0.01:
        st.error("Weights must sum to 1.")
    else:
        with st.spinner("Generating map..."):
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

            # Get lat/lon
            top_lots = top_lots.to_crs(epsg=3857)  # project to meters
            top_lots["lon"] = top_lots.geometry.centroid.to_crs(epsg=4326).x
            top_lots["lat"] = top_lots.geometry.centroid.to_crs(epsg=4326).y

            # Center the map
            center = [top_lots["lat"].mean(), top_lots["lon"].mean()]

            # Normalize final scores for consistent coloring and size scaling
            min_score = top_lots["final_score"].min()
            max_score = top_lots["final_score"].max()
            score_range = max_score - min_score
            top_lots["normalized_score"] = (top_lots["final_score"] - min_score) / score_range

            def get_rank_color(rank):
                if rank <= 100:
                    return [27, 94, 32]  # Dark green
                elif rank <= 200:
                    return [56, 142, 60]
                elif rank <= 300:
                    return [102, 187, 106]
                elif rank <= 400:
                    return [165, 214, 167]
                else:
                    return [232, 245, 233]  # Very light green


            top_lots["color"] = top_lots["rank"].apply(get_rank_color)

            # Prepare pydeck layer
            scatter = pdk.Layer(
                "ScatterplotLayer",
                data=top_lots,
                get_position=["lon", "lat"],
                get_radius=30,  # Reduced from 300 to 30 for less overlap
                get_fill_color="color",
                pickable=True,
                radius_min_pixels=3,
                radius_max_pixels=10,
                auto_highlight=True,
            )

            # View state
            view_state = pdk.ViewState(
                latitude=center[0],
                longitude=center[1],
                zoom=13,
                pitch=0,
            )

            # Tooltip with full score
            tooltip = {
                "html": "<b>Rank:</b> {rank}<br><b>Score:</b> {final_score}",
                "style": {"backgroundColor": "white", "color": "#4a6240", "fontSize": "14px"},
            }


            # Show map
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[scatter],
                tooltip=tooltip,
            ), height=585)

        st.markdown("""
            <style>
            .rank-legend {
                position: fixed;
                top: 110px;  /* ⬇ Lowered from 100px to avoid zoom buttons */
                right: 100px;  /* ⬅ Pulls it slightly inward for alignment */
                background: rgba(255, 255, 255, 0.95);
                padding: 12px 16px;
                border: 1px solid #ccc;
                border-radius: 10px;
                z-index: 9999;
                font-size: 13px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                font-family: sans-serif;
            }
            .rank-legend div {
                display: flex;
                align-items: center;
                margin-bottom: 6px;
            }
            .rank-legend span {
                display: inline-block;
                width: 14px;
                height: 14px;
                margin-right: 8px;
                border-radius: 3px;
                border: 1px solid #999;
            }
            </style>

            <div class="rank-legend">
            <strong>Rank Color Legend</strong><br><br>
            <div><span style='background-color: rgb(27, 94, 32);'></span>Rank 1–100</div>
            <div><span style='background-color: rgb(56, 142, 60);'></span>Rank 101–200</div>
            <div><span style='background-color: rgb(102, 187, 106);'></span>Rank 201–300</div>
            <div><span style='background-color: rgb(165, 214, 167);'></span>Rank 301–400</div>
            <div><span style='background-color: rgb(232, 245, 233);'></span>Rank 401–500</div>
            </div>
            """, unsafe_allow_html=True)

        

        st.session_state.update = False
# -----------------------------
# 📊 Tab 2: Data
# -----------------------------
with tab2:
    if "top_lots" in st.session_state:
        top_lots = st.session_state.top_lots
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
# 📄 Tab 3: Histogram
# -----------------------------
# with tab3:
#     col1, col2 = st.columns([1.5, 0.5])  # Adjusting width ratio

#     with col1:
#         st.markdown("### Score Distribution of Top 500 Sites")
#         with st.container():
#             fig, ax = plt.subplots()
#             ax.hist(top_lots["final_score"], bins=30, color="#4a6240", edgecolor="black")
#             ax.set_xlabel("Final Score")
#             ax.set_ylabel("Frequency")
#             st.pyplot(fig)

with tab3:
    if "top_lots" in st.session_state:
        top_lots = st.session_state.top_lots
        col1, col2 = st.columns([1.5, 0.5])

        with col1:
            st.markdown("### Score Distribution of Top 500 Sites")
            fig, ax = plt.subplots()
            ax.hist(top_lots["final_score"], bins=30, color="#4a6240", edgecolor="black")
            ax.set_xlabel("Final Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
    else:
        st.info("Generate map first by clicking 'Create Map'.")
