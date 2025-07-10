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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Map", "Data","Gallery", "About", "Participatory"])
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
                # score_col = 1 / (1 + c[col])
                norm_col = min_max_normalize(c[col])
                norm_scores.append(norm_col)

            c["final_score"] = sum(w * s for w, s in zip(weight_array, norm_scores))
            ranked = c.sort_values("final_score", ascending=True).reset_index(drop=True)
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

        st.markdown("## Data")

        st.markdown("### How Scores Are Calculated")
        st.markdown("""
        Each lot is evaluated by its distance to critical services like transit, housing, and water infrastructure.

        We **normalize** these distances to a 0–1 scale, and then **sum them** using the weights you assign.  
        This means:

        - **Closer = lower score = better**
        - **Smaller total distance** across all weighted features = **higher ranked site**

        You can download the full ranked list, see how each feature contributed, and explore visuals below.
        """)

        # Add weights as separate columns
        weight_cols = [f"weight_{col}" for col in score_cols]
        for col in score_cols:
            top_lots[f"weight_{col}"] = st.session_state.weights.get(col, 0)

        # Add lat/lon if not present
        if "lat" not in top_lots.columns or "lon" not in top_lots.columns:
            top_lots["lon"] = top_lots.geometry.centroid.to_crs(epsg=4326).x
            top_lots["lat"] = top_lots.geometry.centroid.to_crs(epsg=4326).y

        # GRAPHS

        # Display top 5 ranked sites
        st.markdown("### Top 5 Ranked Sites")
        st.dataframe(top_lots[["id", "rank", "final_score", "lon", "lat"] + score_cols].head())

        # Histogram – Top 50 Ranked Sites
        st.markdown("### Score Distribution – Top 50 Ranked Sites")
        fig1, ax1 = plt.subplots(figsize=(5, 2.2))
        ax1.hist(top_lots["final_score"].head(50), bins=15, color="#4a6240", edgecolor="white")
        ax1.set_xlabel("Final Score", fontsize=9)
        ax1.set_ylabel("Sites", fontsize=9)
        ax1.set_title("Top 50 Score Histogram", fontsize=10)
        ax1.tick_params(axis='both', labelsize=8)
        st.pyplot(fig1)

        # Scatterplot – Final Score vs. Rank
        st.markdown("### Final Score vs. Rank")
        fig2, ax2 = plt.subplots(figsize=(5, 2.2))
        ax2.scatter(top_lots["rank"], top_lots["final_score"], color="#4a6240", alpha=0.8)
        ax2.set_xlabel("Rank", fontsize=9)
        ax2.set_ylabel("Final Score", fontsize=9)
        ax2.set_title("Score by Rank (Top 500)", fontsize=10)
        ax2.tick_params(axis='both', labelsize=8)
        st.pyplot(fig2)

        # Histogram – All 500 Ranked Sites
        st.markdown("### Score Distribution – All 500 Ranked Sites")
        fig3, ax3 = plt.subplots(figsize=(5, 2.2))
        ax3.hist(top_lots["final_score"], bins=30, color="#4a6240", edgecolor="white")
        ax3.set_xlabel("Final Score", fontsize=9)
        ax3.set_ylabel("Sites", fontsize=9)
        ax3.set_title("Histogram of All Final Scores", fontsize=10)
        ax3.tick_params(axis='both', labelsize=8)
        st.pyplot(fig3)


        #download button for top 500 ranked sites
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
# 🖼️ Tab 3: Gallery

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
        for i in range(0, len(entries), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j >= len(entries):
                    break
                entry = entries[i + j]

                with cols[j]:
                    # User Info
                    st.markdown(f"""
                    **Name:** {entry['name']}  
                    **Job:** {entry['occupation']}  
                    **Area:** {entry['location']}  
                    """)

                    st.markdown("**Weights Used:**")

                    # Display weights as a readable table
                    raw_weights = entry.get("weights", {})
                    if raw_weights:
                        formatted_weights = {
                            k.replace("_dist", "").replace("_", " ").title(): v
                            for k, v in raw_weights.items()
                        }
                        df = pd.DataFrame(
                            list(formatted_weights.items()),
                            columns=["Feature", "Weight"]
                        )
                        st.dataframe(
                            df.style.format({"Weight": "{:.2f}"}),
                            use_container_width=True,
                            hide_index=True
                        )

                    # Separator
                    st.markdown("---")


# -----------------------------
with tab4:

    st.markdown("## About This Tool")
    st.image("verticalParklet.png")
    st.markdown(
        "<div style='padding-bottom: 10px; font-size: 0.75rem; color: gray; text-align: center;'>" \
        "Figure 1. Example of vertically spaced tiny home in parking area."
        "</div>",
        unsafe_allow_html=True
    )
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

# -----------------------------
# 🌍️ Tab 5: Participatory (AHP)
# -----------------------------
with tab5:
    st.markdown("## Help Us Understand What Matters Most")
    st.markdown("""
    In this section, you can participate in shaping how we evaluate ideal sites for tiny homes in Oakland.

    We'll show you a few pairs of features. Just tell us which one you think is **more important** and **how much more**.

    Your responses will be used to compute weights — and help inform future planning.
    """)

    # Feature names (short display names, linked to score_cols)
    feature_map = {
        "Transit Access": "transit_dist",
        "Proximity to Housing Services": "public_housing_dist",
        "Access to Water Infrastructure": "water_infrastructure_dist",
        "Near City Facilities": "city_facility_dist",
        "Proximity to Mobile Vending": "mobile_vending_dist",
        "Assisted Housing Nearby": "assisted_housing_dist",
        "Homeless Services": "homeless_service_dist",
        "Public Infrastructure": "general_plan_dist"
    }

    features = list(feature_map.keys())

    # Generate all unique pairwise combinations (AHP style)
    import itertools
    pairs = list(itertools.combinations(features, 2))

    if "comparisons" not in st.session_state:
        st.session_state.comparisons = {}

    st.markdown("### Pairwise Comparisons")

    for f1, f2 in pairs[:5]:  # Limit to 5 for now to reduce overload
        key = f"{f1}__vs__{f2}"
        st.session_state.comparisons[key] = st.select_slider(
            f"How much more important is **{f1}** compared to **{f2}**?",
            options=[
                f"{f1} much more", f"{f1} more", "Equal", f"{f2} more", f"{f2} much more"
            ],
            key=key
        )

    # Button to compute weights
    if st.button("Compute My Priorities"):
        import numpy as np

        size = len(features)
        ahp_matrix = np.ones((size, size))
        for (i, f1) in enumerate(features):
            for (j, f2) in enumerate(features):
                if i >= j:
                    continue
                key = f"{f1}__vs__{f2}"
                if key not in st.session_state.comparisons:
                    continue
                val = st.session_state.comparisons[key]
                scale = {
                    f"{f1} much more": 5,
                    f"{f1} more": 3,
                    "Equal": 1,
                    f"{f2} more": 1/3,
                    f"{f2} much more": 1/5
                }[val]
                ahp_matrix[i][j] = scale
                ahp_matrix[j][i] = 1 / scale

        # Compute normalized principal eigenvector (AHP weights)
        eigvals, eigvecs = np.linalg.eig(ahp_matrix)
        max_index = np.argmax(eigvals)
        weights = np.real(eigvecs[:, max_index])
        weights = weights / weights.sum()

        # Show results
        st.success("Here's what you care about most:")
        st.bar_chart({features[i]: weights[i] for i in range(len(features))})

        # Map back to score_cols
        mapped_weights = {
            feature_map[features[i]]: float(round(weights[i], 4))
            for i in range(len(features))
        }

        # Store for export
        st.session_state.ahp_weights = mapped_weights

        # Download as JSON or CSV
        import io
        import pandas as pd
        csv = pd.DataFrame([mapped_weights]).to_csv(index=False)
        st.download_button("Download My Weights (CSV)", data=csv, file_name="my_ahp_weights.csv")

