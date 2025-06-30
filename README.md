# 🏠 Oakland Tiny Home Site Selector

An interactive web app to **rank and map vacant lots** in Oakland, CA for tiny home development using customizable geospatial criteria.

## 🚀 Try It

**Live site**: 
Current Link [https://tiny-home-backend.streamlit.app/](https://tiny-home-backend.streamlit.app/)

Previous Link [https://tinyhome-backend.onrender.com/](https://tinyhome-backend.onrender.com/)

## 🔍 What It Does

- Lets users **adjust weights** for 10+ urban planning features (e.g., distance to shelters, transit, housing)
- Ranks 500+ sites using **inverse distance scoring**
- Visualizes results in real time with **interactive Pydeck map**
- Provides histogram + CSV download of scores
- Includes a **Gallery tab** to explore shared submissions (powered by Supabase)

## 💻 Tech Stack

- **Frontend**: Streamlit, Pydeck (Mapbox), custom CSS
- **Backend**: Python, Supabase (PostgreSQL + Storage)
- **Data**: GeoPandas, normalized CSV/GeoJSON scoring

## 📦 Features

- 📊 Weight sliders + data sync
- 🗺️ Live-updating map & tooltip
- 🖼️ Supabase-powered map gallery (name, job, neighborhood, weights)
- 🔒 RLS-secure uploads with cloud storage

## 🌱 Future Plans

- Export map as image
- Filter gallery by traits
- Add logins, saved maps, clustering

## 👨‍🔬 Built by

**Kalyan Lab @ UBC** — empowering data-driven urban equity solutions.

## 🧑‍💻 Run Locally

Follow these steps to set up and launch the app on your machine:

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/oakland-tiny-homes.git
cd oakland-tiny-homes
```

### 2. Set Up a Virtual Environment
```bash
python -m venv .venv
```

### 3. Activate it:

- **Windows**:
  ```bash
  .venv\Scripts\activate
  ```

- **Mac/Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```
- **If requirements.txt is missing:**:
  ```bash
    pip install streamlit geopandas matplotlib pydeck pandas supabase
  ```

### 5. Add GeoJSON Data
Ensure the following file is in the root folder:
```bash
candidates_with_features.geojson
```
This contains the geospatial data needed for scoring and mapping.

### 6. Run the App
```bash
streamlit run app.py
```
Replace app.py with your actual script name if different.

