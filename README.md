# 🏠 Oakland Tiny Home Site Selector

An interactive web app to **rank and map vacant lots** in Oakland, CA for tiny home development using customizable geospatial criteria.

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

## 🚀 Try It

**Live site**: [tiny-home-backend.streamlit.app](https://tiny-home-backend.streamlit.app)

## 🌱 Future Plans

- Export map as image
- Filter gallery by traits
- Add logins, saved maps, clustering

## 👨‍🔬 Built by

**Kalyan Lab @ UBC** — empowering data-driven urban equity solutions.
