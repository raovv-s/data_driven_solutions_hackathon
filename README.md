# GeoTarget AI

GeoTarget AI is a geospatial analysis tool that helps you find optimal business locations using **H3 hexagons** and **OpenStreetMap (OSM)** data.

At a high level, the app:
- Builds an H3 hex grid around a user-selected latitude/longitude.
- Pulls relevant features from OSM (for example, businesses tagged with `amenity`).
- Scores nearby hexagons by feature density, then visualizes the best candidates on an interactive map.

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the UI

```bash
streamlit run app.py
```

Then open the URL shown in your terminal.

## Notes

- The first run may take a little longer because OSMnx fetches data from OpenStreetMap.
- The scoring logic in `src/engine.py` is intentionally simple as a starter scaffold; you can extend it with weights, normalization, or additional metrics.

