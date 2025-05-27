import streamlit as st
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import pandas as pd
from urllib.parse import parse_qs
import os
import json
from supabase import create_client, Client
import uuid
from io import BytesIO
import streamlit.components.v1 as components
from streamlit.components.v1 import html


# -----------------------------

# setting up Supabase client
SUPABASE_URL = "https://sjsgkndenvtzgjihermn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNqc2drbmRlbnZ0emdqaWhlcm1uIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDgzMjM5MjAsImV4cCI6MjA2Mzg5OTkyMH0.AdEST3BzTwIuzcMfWmPZfRZPf4aNhC2xQG8vVWCks50"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------PAGE CONFIG ----------
# this needs to be at the top of the file

#trying to make entire layout full width
#removes default width cap, makes all tabs span full width
st.set_page_config(
    page_title = "Oakland Tiny Home", 
    page_icon = "tinyhomefavicon.png",
    layout = "wide")

# -----------------------------

#getting the height of the page
st.markdown("""
<script>
const params = new URLSearchParams(window.location.search);
if (!params.has("vh")) {
    params.set("vh", window.innerHeight);
    window.location.search = params.toString();
}
</script>
""", unsafe_allow_html=True)

vh_param = st.query_params.get("vh", ["700"])[0]
try:
    vh = int(vh_param)
except ValueError:
    vh = 700  # fallback


map_height = int(0.75 * vh)      # 75% of screen for map
hist_height = int(0.7 * vh / 100)  # inches for matplotlib (approx 100px/in)



# -----------------------------
#trying to add lightweight warm-up endpoint
# Optional warm-up support (advanced use, only for backend triggers)
query_params = st.query_params
if query_params.get("warmup", ["false"])[0].lower() == "true":
    _ = load_candidates()
    st.write("Warmed up!")
    st.stop()
# -----------------------------
# 💴 UI Style
# -----------------------------

st.markdown("""
    <style>
    html, body, .main, .block-container {
            height: 100vh;
            margin: 0;
            padding: 0;
            overflow-y: auto;
    }
    # making sure tabs contanier/inner tab container take up full height
    
    section [data-testid="stTab"] > div{
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    </style>
""", unsafe_allow_html=True)
            

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
tab1, tab2, tab3, tab4 = st.tabs(["Map", "Data","Gallery", "About"])
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

    if st.sidebar.button("Create"):
        st.session_state.update = True

    #title and instructions block
    st.markdown("## Ranked Tiny Home Site Map")
    st.markdown("""
        This interactive map displays the **top 500 ranked vacant lots** in Oakland, CA 
        that may be suitable for tiny home development. You can:

        - Adjust the weights of various different criteria using the sidebar.
        - Click "Create Map" to regenerate rankings.
        - Hover over any circle to see the site's rank and score.
        - Darker green = higher ranking (Rank 1–100); lighter = lower (Rank 401–500).
        - Use the **"Data" tab** to view a table of the top 5 sites and download all 500 as CSV.
        - View the **"Histogram" tab** to explore the score distribution.
        """)


    #map 
    if not st.session_state.update:
        st.info("Adjust weights and click 'Create' to generate site rankings.")
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
            st.session_state.top_lots = top_lots

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
                #could customize to be a different style if necessary, like if we want terrain. 
                # but since we have data overlay this light version is a good option
                initial_view_state=view_state,
                layers=[scatter],
                tooltip=tooltip,
            ), height=map_height)

            # New legend code
            st.markdown("""
            <style>
            .rank-legend-bar {
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-start;
                gap: 1rem;
                margin-top: 1rem;
                padding: 1rem;
                background: #f5f5f5;
                border-radius: 8px;
                font-size: 14px;
                font-family: sans-serif;
                border: 1px solid #ddd;
            }

            .rank-item {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .rank-box {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #999;
            }
            </style>

            <div class="rank-legend-bar">
            <div class="rank-item"><div class="rank-box" style="background-color: rgb(27, 94, 32);"></div>Rank 1–100</div>
            <div class="rank-item"><div class="rank-box" style="background-color: rgb(56, 142, 60);"></div>Rank 101–200</div>
            <div class="rank-item"><div class="rank-box" style="background-color: rgb(102, 187, 106);"></div>Rank 201–300</div>
            <div class="rank-item"><div class="rank-box" style="background-color: rgb(165, 214, 167);"></div>Rank 301–400</div>
            <div class="rank-item"><div class="rank-box" style="background-color: rgb(232, 245, 233);"></div>Rank 401–500</div>
            </div>

            <br>
            """, unsafe_allow_html=True)


            # Optionally save and allow download of HTML snapshot of map
            deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[scatter],
                tooltip=tooltip,
            )


            # Save to HTML file
            deck.to_html("latest_map.html")

            # Allow download of static HTML map
            with open("latest_map.html", "rb") as f:
                st.download_button("Download Map as HTML Snapshot", f.read(), file_name="map_snapshot.html")


        

        st.session_state.update = False
# -----------------------------
# 📊 Tab 2: Data
# -----------------------------
with tab2:
    if "top_lots" in st.session_state:
        top_lots = st.session_state.top_lots.copy()

        # Add weights as separate columns
        weight_cols = [f"weight_{col}" for col in score_cols]
        for col in score_cols:
            top_lots[f"weight_{col}"] = st.session_state.weights.get(col, 0)

        # Add lat/lon if not present
        if "lat" not in top_lots.columns or "lon" not in top_lots.columns:
            top_lots["lon"] = top_lots.geometry.centroid.to_crs(epsg=4326).x
            top_lots["lat"] = top_lots.geometry.centroid.to_crs(epsg=4326).y

        st.markdown("### Top 5 Ranked Sites")
        st.dataframe(top_lots[["id", "rank", "final_score", "lon", "lat"] + score_cols].head())


        # Histogram of final scores 500 sites
        st.markdown("### Score Distribution of Top 500 Sites")
        fig, ax = plt.subplots(figsize=(10, hist_height))
        ax.hist(top_lots["final_score"], bins=30, color="#4a6240", edgecolor="black")
        ax.set_xlabel("Final Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)  

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
# 🖼️ Tab 4: Gallery
# -----------------------------
# -----------------------------
# 🖼️ Tab 3: Gallery

import streamlit.components.v1 as components

with tab3:
    st.markdown("## Gallery: Explore Shared Tiny Home Maps")
    st.markdown("""
    Welcome to the community gallery!

    Here, you can see what other users think the best locations for tiny homes are — based on their personal priorities. 
    View their slider weights, read a bit about them, and explore their generated map!
    """)

    with st.expander("Upload Your Map"):
        with st.form("supabase_upload", clear_on_submit=True):
            name = st.text_input("Your Name")
            occupation = st.text_input("Occupation")
            location = st.text_input("City / Area in Oakland")
            uploaded_file = st.file_uploader("Upload CSV file of your map", type=["csv"])
            submit = st.form_submit_button("Submit")

            if submit:
                if uploaded_file and name and location:
                    df = pd.read_csv(uploaded_file)
                    weights = {
                        col.replace("weight_", ""): round(df[col].iloc[0], 3)
                        for col in df.columns if col.startswith("weight_")
                    }
                    file_id = str(uuid.uuid4())
                    file_name = f"{file_id}.csv"
                    temp_file_path = os.path.join(os.getcwd(), file_name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    storage_response = supabase.storage.from_('maps').upload(
                        path=file_name,
                        file=temp_file_path,
                        file_options={"content-type": "text/csv"}
                    )

                    if not storage_response:
                        st.error("❌ No response from Supabase upload.")
                    elif hasattr(storage_response, "data") and storage_response.data is None:
                        st.error("❌ Upload may have failed: no data returned.")
                    else:
                        file_url = f"{SUPABASE_URL}/storage/v1/object/public/maps/{file_name}"
                        data = {
                            "name": name,
                            "occupation": occupation,
                            "location": location,
                            "weights": weights,
                            "file_url": file_url
                        }
                        res = supabase.table("submissions").insert(data).execute()
                        if res.data:
                            st.success("Submission uploaded successfully!")
                        else:
                            st.error("❌ Something went wrong — insert failed.")
                else:
                    st.warning("Please complete all fields and upload a file.")

    st.markdown("### Shared Maps")

    try:
        submissions = supabase.table("submissions").select("*").order("created_at", desc=True).execute()
        entries = submissions.data
    except Exception as e:
        st.error(f"Failed to load gallery: {e}")
        entries = []

    if not entries:
        st.info("No gallery submissions yet.")
    else:
        cols = st.columns(3)
        for i, entry in enumerate(entries):
            col = cols[i % 3]

            with col:
                weights = entry.get("weights", {})
                pretty_weights = {
                    k.replace("_dist", "").replace("_", " ").title(): v
                    for k, v in weights.items()
                }
                weight_df = pd.DataFrame(list(pretty_weights.items()), columns=["Feature", "Weight"])
                weight_table_html = weight_df.to_html(index=False)

                html = f"""
                <div style="
                    background-color: #f9fafb;
                    border: 1px solid #ddd;
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 16px;
                    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
                    font-family: sans-serif;
                ">
                    <p><strong>Name:</strong> {entry['name']}</p>
                    <p><strong>Job:</strong> {entry['occupation']}</p>
                    <p><strong>Area:</strong> {entry['location']}</p>
                    <p><strong>Weights Used:</strong></p>
                    {weight_table_html}
                </div>
                """
                from streamlit.components.v1 import html

from streamlit.components.v1 import html

for i in range(0, len(entries), 3):
    row = st.columns(3)
    for j in range(3):
        if i + j >= len(entries):
            break
        entry = entries[i + j]
        name = entry["name"]
        occupation = entry["occupation"]
        location = entry["location"]
        weights = entry.get("weights", {})

        # HTML Table rows
        table_rows = "".join(
            f"<tr><td>{k.replace('_dist', '').replace('_', ' ').title()}</td><td style='text-align:right;'>{v:.2f}</td></tr>"
            for k, v in weights.items()
        )

        # Single card block
        card_html = f"""
        <div style="
            font-family: 'Segoe UI', sans-serif;
            background-color: #ffffff;
            border: 1px solid #e6e6e6;
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 20px;
        ">
            <p style="margin: 4px 0;"><strong>Name:</strong> {name}</p>
            <p style="margin: 4px 0;"><strong>Job:</strong> {occupation}</p>
            <p style="margin: 8px 0;"><strong>Area:</strong> {location}</p>

            <p style="margin-top: 14px; font-weight: 600;">Weights Used:</p>
            <table style="
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
                margin-top: 6px;
            ">
                <thead>
                    <tr style="border-bottom: 1px solid #e0e0e0;">
                        <th style="text-align:left; padding: 4px;">Feature</th>
                        <th style="text-align:right; padding: 4px;">Weight</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """
        with row[j]:
            html(card_html, height=270)




# -----------------------------
with tab4:

    st.markdown("## About This Tool")
    st.write("""
    This tool helps identify and rank vacant lots in Oakland, California
    for potential tiny home development. Using multiple spatial
    criteria, such as proximity to homeless services, transit, and
    housing infrastructure. Users can customize weights to explore
    optimal locations interactively.

    ### Features:
    - various customizable criteria via sliders
    - Interactive map with ranked 500 top sites
    - Downloadable data and score histogram
    - Clean, responsive design using Streamlit and Pydeck

    Built by Kalyan Lab at UBC with ❤️ for equitable urban planning.
    """)
