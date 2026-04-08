from __future__ import annotations

import json
from typing import Any

import pandas as pd
import geopandas as gpd
import streamlit as st
from folium import GeoJson

from src.ai_analyst import generate_business_report
from src.data_loader import generate_walk_isochrones, get_building_data, load_osm_features, simulate_competitor_ratings
from src.engine import find_optimal_locations

try:
    from streamlit_folium import st_folium
except Exception:  # pragma: no cover
    st_folium = None

import folium

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover
    go = None


def _gdf_to_geojson(gdf: gpd.GeoDataFrame) -> dict:
    return json.loads(gdf.to_json())


def _ensure_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Make geometries point-based for spatial joins."""
    if gdf.empty:
        return gdf
    geom_type = gdf.geometry.geom_type
    if geom_type.iloc[0] == "Point":
        return gdf
    return gdf.copy().assign(geometry=gdf.geometry.representative_point())


def _format_magnet(m: pd.Series) -> str:
    geom = m.get("geometry")
    if geom is None:
        return "Unknown magnet"
    # geometry is expected to be a Point in EPSG:4326
    name = m.get("name") if "name" in m else None
    lat = getattr(geom, "y", None)
    lon = getattr(geom, "x", None)
    if name:
        return f"{name} ({lat:.5f}, {lon:.5f})"
    return f"({lat:.5f}, {lon:.5f})"


def _generate_smart_insights(
    top_row: pd.Series,
    target_business: str,
    magnet_types: list[str],
) -> str:
    score = float(top_row.get("score", 0.0))
    magnet_count = float(top_row.get("magnet_count", 0.0))
    competitor_count = float(top_row.get("competitor_count", 0.0))

    magnets_label = ", ".join(magnet_types) if magnet_types else "nearby magnets"

    if magnet_count <= 0:
        magnet_msg = f"it has no nearby magnets detected (magnet_count={magnet_count:.0f})."
    else:
        magnet_msg = f"it has a high density of magnets ({magnet_count:.0f} nearby) from: {magnets_label}."

    if competitor_count <= 0:
        competition_msg = f"Competition is very low (competitor_count={competitor_count:.0f})."
    else:
        competition_msg = f"Competition exists but is limited (competitor_count={competitor_count:.0f})."

    return f"Recommendation: This area is strong for a {target_business} because {magnet_msg} {competition_msg} Overall score: {score:.1f}/10."


def _market_sentiment_badge(avg_rating: float | None, competitor_count: int) -> tuple[str, str]:
    """
    Return (label, color) for market sentiment.
    """
    if competitor_count <= 0:
        return ("No Competition", "#6c757d")
    if avg_rating is None:
        return ("Unknown", "#6c757d")
    if avg_rating < 3.8:
        return ("Disappointed Customers", "#dc3545")
    if avg_rating >= 4.3:
        return ("High Loyalty", "#198754")
    return ("Mixed Sentiment", "#ffc107")


def _build_tag_specs(magnet_types: list[str]) -> list[dict[str, Any]]:
    """Map UI magnet types to OSM tag filters for loading features."""
    magnet_tag_specs_map: dict[str, list[dict]] = {
        "Subway entrances": [{"railway": "subway_entrance"}, {"subway": "entrance"}, {"station": "subway"}],
        "Bus stops": [{"highway": "bus_stop"}],
        "Universities": [{"amenity": "university"}],
    }
    return [spec for mt in magnet_types for spec in magnet_tag_specs_map[mt]]


def _compute_best_hex_metrics(
    *,
    lat: float,
    lon: float,
    radius_km: float,
    resolution: int,
    target_business: str,
    magnet_types: list[str],
    focus_under_served: bool,
) -> dict[str, float] | None:
    """Compute the best (highest score) hexagon metrics for a given point."""
    target_tags_map = {
        "Cafe": {"amenity": "cafe"},
        "Restaurant": {"amenity": "restaurant"},
        "Bar": {"amenity": "bar"},
    }

    competitor_tags = target_tags_map[target_business]
    magnet_tag_specs = _build_tag_specs(magnet_types)

    competitor_features = load_osm_features(
        lat=lat,
        lon=lon,
        distance_m=int(radius_km * 1000),
        tags=competitor_tags,
    )

    magnet_features_parts: list[gpd.GeoDataFrame] = []
    for tags in magnet_tag_specs:
        magnet_features_parts.append(
            load_osm_features(
                lat=lat,
                lon=lon,
                distance_m=int(radius_km * 1000),
                tags=tags,
            )
        )

    magnet_features = (
        gpd.GeoDataFrame(pd.concat(magnet_features_parts, ignore_index=True), crs="EPSG:4326")
        if magnet_features_parts
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    )

    residential_buildings = get_building_data(
        center_point=(lat, lon),
        radius_m=int(radius_km * 1000),
    )

    hexes_all = find_optimal_locations(
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        resolution=resolution,
        competitor_features=competitor_features,
        magnet_features=magnet_features,
        residential_buildings=residential_buildings,
        focus_under_served=focus_under_served,
        top_k=None,
    )

    if hexes_all.empty:
        return None

    best = hexes_all.sort_values("score", ascending=False).head(1).iloc[0]
    return {
        "h3_index": float("nan"),  # placeholder; not used for radar
        "score": float(best.get("score", 0.0) or 0.0),
        "magnet_count": float(best.get("magnet_count", 0.0) or 0.0),
        "competitor_count": float(best.get("competitor_count", 0.0) or 0.0),
        "gap_score": float(best.get("gap_score", 0.0) or 0.0),
        "population_proxy": float(best.get("population_proxy", 0.0) or 0.0),
        "raw_score": float(best.get("raw_score", 0.0) or 0.0),
        "geometry": best.get("geometry"),
    }


def _normalize_to_0_10(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    return max(0.0, min(10.0, (value - vmin) / (vmax - vmin) * 10.0))


def _render_radar_chart(point_a: dict[str, float], point_b: dict[str, float], label_a: str, label_b: str) -> None:
    if go is None:
        st.warning("Radar chart needs `plotly` installed.")
        return

    # Strengths (higher is better):
    a_magnets = float(point_a.get("magnet_count", 0.0) or 0.0)
    b_magnets = float(point_b.get("magnet_count", 0.0) or 0.0)

    a_pop = float(point_a.get("population_proxy", 0.0) or 0.0)
    b_pop = float(point_b.get("population_proxy", 0.0) or 0.0)

    # Competition is a penalty, so invert it: higher "low competition strength" is better.
    a_comp_strength = max(float(point_a.get("competitor_count", 0.0) or 0.0), float(point_b.get("competitor_count", 0.0) or 0.0)) - float(
        point_a.get("competitor_count", 0.0) or 0.0
    )
    b_comp_strength = max(float(point_a.get("competitor_count", 0.0) or 0.0), float(point_b.get("competitor_count", 0.0) or 0.0)) - float(
        point_b.get("competitor_count", 0.0) or 0.0
    )

    a_score = float(point_a.get("score", 0.0) or 0.0)
    b_score = float(point_b.get("score", 0.0) or 0.0)

    categories = [
        "Overall Score",
        "Transport (Magnets)",
        "Residential Density",
        "Low Competition",
        "Under-served (Gap)",
    ]

    transport_vals = [a_magnets, b_magnets]
    pop_vals = [a_pop, b_pop]
    comp_vals = [a_comp_strength, b_comp_strength]
    score_vals = [a_score, b_score]
    gap_vals = [float(point_a.get("gap_score", 0.0) or 0.0), float(point_b.get("gap_score", 0.0) or 0.0)]

    a_r = [
        _normalize_to_0_10(a_score, min(score_vals), max(score_vals)),
        _normalize_to_0_10(a_magnets, min(transport_vals), max(transport_vals)),
        _normalize_to_0_10(a_pop, min(pop_vals), max(pop_vals)),
        _normalize_to_0_10(a_comp_strength, min(comp_vals), max(comp_vals)),
        _normalize_to_0_10(float(point_a.get("gap_score", 0.0) or 0.0), min(gap_vals), max(gap_vals)),
    ]
    b_r = [
        _normalize_to_0_10(b_score, min(score_vals), max(score_vals)),
        _normalize_to_0_10(b_magnets, min(transport_vals), max(transport_vals)),
        _normalize_to_0_10(b_pop, min(pop_vals), max(pop_vals)),
        _normalize_to_0_10(b_comp_strength, min(comp_vals), max(comp_vals)),
        _normalize_to_0_10(float(point_b.get("gap_score", 0.0) or 0.0), min(gap_vals), max(gap_vals)),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=a_r + [a_r[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=label_a,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=b_r + [b_r[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=label_b,
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10], visible=True)),
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="GeoTarget AI", layout="wide")
    st.title("GeoTarget AI")
    st.caption("Find optimal business locations using H3 hexagons and OpenStreetMap data.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Inputs")
        lat = st.number_input("Latitude", value=37.7749, format="%.6f")
        lon = st.number_input("Longitude", value=-122.4194, format="%.6f")
        radius_km = st.slider("Search radius (km)", min_value=0.5, max_value=20.0, value=3.0, step=0.5)
        resolution = st.slider("H3 resolution", min_value=3, max_value=9, value=7, step=1)
        top_k = st.slider("Top hexes to show", min_value=1, max_value=20, value=5, step=1)

        target_business_options = ["Cafe", "Restaurant", "Bar"]
        target_business = st.selectbox("Target Business", target_business_options, index=0)

        magnet_type_options = ["Subway entrances", "Bus stops", "Universities"]
        magnet_types = st.multiselect("Magnet Types", magnet_type_options, default=["Bus stops", "Universities"])

        if not magnet_types:
            st.info("Select at least one Magnet Type.")
            st.stop()

        show_walkability = st.checkbox("Show walkability (5 & 10 min)", value=True)
        focus_under_served = st.checkbox("Focus on Under-served Areas", value=False)

    with col_right:
        tab1, tab2 = st.tabs(["Analyze Location", "Compare Locations"])

        # -----------------------
        # Tab 1: Analyze Location
        # -----------------------
        with tab1:
            competitor_tags = {
                "Cafe": {"amenity": "cafe"},
                "Restaurant": {"amenity": "restaurant"},
                "Bar": {"amenity": "bar"},
            }[target_business]

            magnet_tag_specs = _build_tag_specs(magnet_types)
            competitor_features = load_osm_features(
                lat=lat,
                lon=lon,
                distance_m=int(radius_km * 1000),
                tags=competitor_tags,
            )
            competitor_features = simulate_competitor_ratings(competitor_features)

            magnet_features_parts: list[gpd.GeoDataFrame] = []
            for tags in magnet_tag_specs:
                magnet_features_parts.append(
                    load_osm_features(
                        lat=lat,
                        lon=lon,
                        distance_m=int(radius_km * 1000),
                        tags=tags,
                    )
                )

            magnet_features = (
                gpd.GeoDataFrame(pd.concat(magnet_features_parts, ignore_index=True), crs="EPSG:4326")
                if magnet_features_parts
                else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            )

            residential_buildings = get_building_data(center_point=(lat, lon), radius_m=int(radius_km * 1000))

            hexes_all = find_optimal_locations(
                lat=lat,
                lon=lon,
                radius_km=radius_km,
                resolution=resolution,
                competitor_features=competitor_features,
                magnet_features=magnet_features,
                residential_buildings=residential_buildings,
                focus_under_served=focus_under_served,
                top_k=None,
            )

            if hexes_all.empty:
                st.warning("No hexes generated. Try a different radius or resolution.")
                st.stop()

            # Metric cards
            best_score = float(hexes_all["score"].max())
            # Approximate total area scanned as circle (km^2)
            total_area_km2 = 3.14159 * (radius_km ** 2)
            active_competitors = int(competitor_features.shape[0])

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Best Score Found", f"{best_score:.1f} / 10")
            mcol2.metric("Total Area Scanned", f"{total_area_km2:.1f} km²")
            mcol3.metric("Active Competitors", str(active_competitors))

            st.subheader("Result")

            # Dark mode basemap
            m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB dark_matter")

            if show_walkability:
                with st.spinner("Building walkability boundaries (may take a while)..."):
                    walk_iso = generate_walk_isochrones(lat=lat, lon=lon, walk_minutes=(5, 10))

                if not walk_iso.empty and walk_iso.geometry.notna().any():
                    for _, iso_row in walk_iso.iterrows():
                        minutes = int(iso_row["walk_minutes"])
                        geom = iso_row["geometry"]
                        if geom is None or geom.is_empty:
                            continue
                        color = "#1f77b4" if minutes == 5 else "#2ca02c"
                        row_gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=walk_iso.crs)
                        gj = _gdf_to_geojson(row_gdf)
                        GeoJson(
                            gj,
                            name=f"Walkability: {minutes} min",
                            style_function=lambda feature, c=color: {
                                "fillColor": c,
                                "color": c,
                                "weight": 1,
                                "fillOpacity": 0.18,
                            },
                            tooltip=f"Walkability boundary: {minutes} min",
                        ).add_to(m)

            hexes_display = hexes_all.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
            for _, row in hexes_display.iterrows():
                h3_index = row["h3_index"]
                score = float(row["score"]) if "score" in row else 0.0
                competitor_count = float(row["competitor_count"]) if "competitor_count" in row else 0.0
                magnet_count = float(row["magnet_count"]) if "magnet_count" in row else 0.0
                raw_score = float(row["raw_score"]) if "raw_score" in row else 0.0

                row_gdf = gpd.GeoDataFrame(
                    {
                        "h3_index": [h3_index],
                        "score": [score],
                        "raw_score": [raw_score],
                        "magnet_count": [magnet_count],
                        "competitor_count": [competitor_count],
                        "geometry": [row["geometry"]],
                    },
                    crs=hexes_all.crs,
                )
                gj = _gdf_to_geojson(row_gdf)

                t = score / 10.0 if score is not None else 0.0
                color = "darkgreen" if t >= 0.8 else ("green" if t >= 0.4 else "lightgreen")

                GeoJson(
                    gj,
                    name=f"{h3_index} (score={score:.1f})",
                    style_function=lambda _: {
                        "fillColor": color,
                        "color": "#222222",
                        "weight": 1,
                        "fillOpacity": 0.55,
                    },
                    tooltip=(
                        f"{h3_index}"
                        f"<br/>score: {score:.1f} / 10"
                        f"<br/>magnet_count: {magnet_count:.0f}"
                        f"<br/>competitor_count: {competitor_count:.0f}"
                        f"<br/>gap_score: {float(row.get('gap_score', 0.0) or 0.0):.1f} / 10"
                    ),
                ).add_to(m)

            # Simple POI markers with custom-ish icons (using built-in emojis via HTML)
            # Target business markers
            if not competitor_features.empty:
                comp_pts = _ensure_points(competitor_features)
                for _, r in comp_pts.head(200).iterrows():
                    g = r.geometry
                    if g is None:
                        continue
                    folium.Marker(
                        location=[g.y, g.x],
                        popup=f"{target_business} competitor",
                        icon=folium.DivIcon(
                            html=f'<div style="font-size: 16px;">☕</div>' if target_business == "Cafe" else '<div style="font-size: 16px;">🏬</div>'
                        ),
                    ).add_to(m)

            # Magnet markers (coarsely)
            if not magnet_features.empty:
                mag_pts = _ensure_points(magnet_features)
                for _, r in mag_pts.head(200).iterrows():
                    g = r.geometry
                    if g is None:
                        continue
                    folium.Marker(
                        location=[g.y, g.x],
                        popup="Magnet",
                        icon=folium.DivIcon(html='<div style="font-size: 14px;">🚌</div>'),
                    ).add_to(m)

            # Legend for score colors
            legend_html = """
            <div style="
                position: fixed;
                bottom: 20px;
                left: 20px;
                z-index: 9999;
                background-color: rgba(20, 20, 20, 0.8);
                padding: 10px 14px;
                border-radius: 6px;
                color: #f0f0f0;
                font-size: 12px;
                ">
                <b>Hex Score Legend</b><br>
                <span style="display:inline-block;width:12px;height:12px;background-color:lightgreen;margin-right:6px;"></span>
                0–4: Low Potential<br>
                <span style="display:inline-block;width:12px;height:12px;background-color:green;margin-right:6px;"></span>
                4–8: Medium Potential<br>
                <span style="display:inline-block;width:12px;height:12px;background-color:darkgreen;margin-right:6px;"></span>
                8–10: High Potential
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            clicked_h3 = None
            if st_folium is None:
                st.warning("Install `streamlit-folium` to render the clickable map.")
            else:
                map_data = st_folium(
                    m,
                    width=700,
                    height=520,
                    returned_objects=["last_object_clicked"],
                    key="geotarget_map_analyze",
                )
                clicked_obj = (map_data or {}).get("last_object_clicked")
                if isinstance(clicked_obj, dict):
                    props = clicked_obj.get("properties") or {}
                    clicked_h3 = props.get("h3_index")

            st.subheader("Location Analysis")
            if not clicked_h3:
                st.info("Click a hexagon on the map to see details.")
            else:
                clicked_row = hexes_all.loc[hexes_all["h3_index"] == clicked_h3].head(1)
                if clicked_row.empty:
                    st.warning("Clicked hex not found in current results.")
                else:
                    row = clicked_row.iloc[0]
                    total_score = float(row["score"])
                    competitor_count = float(row["competitor_count"])
                    clicked_geom = row["geometry"]

                    # Average competitor rating inside this hex (simulated).
                    avg_rating = None
                    comp_pts = _ensure_points(competitor_features).to_crs("EPSG:4326")
                    if not comp_pts.empty and "rating" in comp_pts.columns:
                        hex_gdf_c = gpd.GeoDataFrame({"geometry": [clicked_geom]}, crs="EPSG:4326")
                        inside_comp = gpd.sjoin(comp_pts, hex_gdf_c, how="inner", predicate="within")
                        if not inside_comp.empty:
                            avg_rating = float(inside_comp["rating"].astype(float).mean())

                    clicked_payload = {
                        "competitor_count": competitor_count,
                        "magnet_count": float(row.get("magnet_count", 0.0) or 0.0),
                        "population_proxy": float(row.get("population_proxy", 0.0) or 0.0),
                        "score": total_score,
                        "magnet_types": magnet_types,
                        "avg_competitor_rating": avg_rating,
                    }

                    st.markdown(f"**Total Score:** {total_score:.1f} / 10")
                    st.markdown(f"**Nearby competitors:** {competitor_count:.0f}")
                    if avg_rating is not None:
                        st.markdown(f"**Avg competitor rating:** {avg_rating:.2f} / 5.0")

                    label, color = _market_sentiment_badge(avg_rating, int(round(competitor_count)))
                    st.markdown(
                        f"**Market Sentiment:** <span style='background:{color};color:#111;padding:2px 8px;border-radius:10px;'>"
                        f"{label}</span>",
                        unsafe_allow_html=True,
                    )

                    magnets_points = _ensure_points(magnet_features).to_crs("EPSG:4326")
                    hex_gdf = gpd.GeoDataFrame({"geometry": [clicked_geom]}, crs="EPSG:4326")
                    if magnets_points.empty:
                        inside = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
                    else:
                        inside = gpd.sjoin(magnets_points, hex_gdf, how="inner", predicate="within")

                    if inside.empty:
                        st.write("**Closest magnets:** none found in this hex.")
                    else:
                        centroid = clicked_geom.centroid
                        centroid_3857 = gpd.GeoSeries([centroid], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
                        inside_3857 = inside.to_crs("EPSG:3857")
                        dists_m = inside_3857.geometry.distance(centroid_3857)
                        inside = inside.to_crs("EPSG:4326")
                        inside = inside.assign(dist_m=dists_m.values)
                        inside_sorted = inside.sort_values("dist_m").head(3)
                        magnets_list = [_format_magnet(r) for _, r in inside_sorted.iterrows()]
                        st.write("**Closest magnets (top 3):**")
                        st.write(magnets_list)

                    st.subheader("Professional Insight")
                    st.write(generate_business_report(clicked_payload, business_type=target_business))

            st.divider()
            st.subheader("Score Distribution (all hexes)")
            score_series = hexes_all["score"].astype(float)
            bins = pd.cut(score_series, bins=10, include_lowest=True)
            dist = score_series.groupby(bins).size()
            dist.index = dist.index.astype(str)
            dist.index.name = "score_bin"
            dist_df = dist.rename("count").to_frame()
            st.bar_chart(dist_df)

            st.subheader("Top 5 Recommended Locations")
            top5 = hexes_all.sort_values("score", ascending=False).head(5)
            cols_top5 = ["h3_index", "score", "magnet_count", "competitor_count", "gap_score", "population_proxy"]
            cols_top5 = [c for c in cols_top5 if c in top5.columns]
            st.dataframe(top5[cols_top5], use_container_width=True)

            best_hex = hexes_all.sort_values("score", ascending=False).head(1).iloc[0]
            st.subheader("Smart Insights")
            st.write(_generate_smart_insights(best_hex, target_business=target_business, magnet_types=magnet_types))

            st.subheader("Download Report")
            report_df = hexes_all.copy()
            keep_cols = [c for c in ["h3_index", "score", "raw_score", "magnet_count", "competitor_count", "population_proxy"] if c in report_df.columns]
            if "gap_score" in report_df.columns:
                keep_cols.insert(keep_cols.index("competitor_count") + 1, "gap_score")
            report_df = report_df[keep_cols]
            csv_bytes = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Report (CSV)",
                data=csv_bytes,
                file_name="geotarget_ai_report.csv",
                mime="text/csv",
            )

        # -----------------------
        # Tab 2: Compare Locations
        # -----------------------
        with tab2:
            st.subheader("Scenario Comparison")

            if st_folium is None:
                st.warning("Install `streamlit-folium` to enable point picking.")
            else:
                # Initialize session state for comparison points.
                if "compare_point_a" not in st.session_state:
                    st.session_state.compare_point_a = None
                if "compare_point_b" not in st.session_state:
                    st.session_state.compare_point_b = None

                st.write("Click once to set **Point A**, click again to set **Point B**.")

                m2 = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB dark_matter")

                if st.session_state.compare_point_a is not None:
                    folium.CircleMarker(
                        location=[st.session_state.compare_point_a["lat"], st.session_state.compare_point_a["lng"]],
                        radius=7,
                        color="blue",
                        fill=True,
                        fillOpacity=0.9,
                        popup="Point A",
                    ).add_to(m2)
                if st.session_state.compare_point_b is not None:
                    folium.CircleMarker(
                        location=[st.session_state.compare_point_b["lat"], st.session_state.compare_point_b["lng"]],
                        radius=7,
                        color="red",
                        fill=True,
                        fillOpacity=0.9,
                        popup="Point B",
                    ).add_to(m2)

                map_data2 = st_folium(
                    m2,
                    width=700,
                    height=520,
                    returned_objects=["last_clicked"],
                    key="geotarget_map_compare",
                )
                clicked2 = (map_data2 or {}).get("last_clicked")

                if isinstance(clicked2, dict) and "lat" in clicked2 and "lng" in clicked2:
                    new_point = {"lat": float(clicked2["lat"]), "lng": float(clicked2["lng"])}

                    if st.session_state.compare_point_a is None:
                        st.session_state.compare_point_a = new_point
                    elif st.session_state.compare_point_b is None:
                        # Ensure A and B differ.
                        a = st.session_state.compare_point_a
                        if abs(a["lat"] - new_point["lat"]) > 1e-6 or abs(a["lng"] - new_point["lng"]) > 1e-6:
                            st.session_state.compare_point_b = new_point
                    else:
                        # Update B on further clicks.
                        st.session_state.compare_point_b = new_point

                cA = st.session_state.compare_point_a
                cB = st.session_state.compare_point_b

                if st.button("Clear Points"):
                    st.session_state.compare_point_a = None
                    st.session_state.compare_point_b = None
                    st.rerun()

                if cA is None or cB is None:
                    st.info("Set both Point A and Point B to see comparison.")
                else:
                    with st.spinner("Computing scenario comparison..."):
                        metrics_a = _compute_best_hex_metrics(
                            lat=cA["lat"],
                            lon=cA["lng"],
                            radius_km=radius_km,
                            resolution=resolution,
                            target_business=target_business,
                            magnet_types=magnet_types,
                            focus_under_served=focus_under_served,
                        )
                        metrics_b = _compute_best_hex_metrics(
                            lat=cB["lat"],
                            lon=cB["lng"],
                            radius_km=radius_km,
                            resolution=resolution,
                            target_business=target_business,
                            magnet_types=magnet_types,
                            focus_under_served=focus_under_served,
                        )

                    if metrics_a is None or metrics_b is None:
                        st.warning("Could not compute one of the locations. Try different points.")
                    else:
                        score_a = float(metrics_a.get("score", 0.0) or 0.0)
                        score_b = float(metrics_b.get("score", 0.0) or 0.0)
                        comp_a = float(metrics_a.get("competitor_count", 0.0) or 0.0)
                        comp_b = float(metrics_b.get("competitor_count", 0.0) or 0.0)
                        mag_a = float(metrics_a.get("magnet_count", 0.0) or 0.0)
                        mag_b = float(metrics_b.get("magnet_count", 0.0) or 0.0)
                        pop_a = float(metrics_a.get("population_proxy", 0.0) or 0.0)
                        pop_b = float(metrics_b.get("population_proxy", 0.0) or 0.0)
                        gap_a = float(metrics_a.get("gap_score", 0.0) or 0.0)
                        gap_b = float(metrics_b.get("gap_score", 0.0) or 0.0)

                        comp_table = pd.DataFrame(
                            {
                                "Metric": ["Score", "Gap Score", "Competitors", "Magnets", "Population Proxy"],
                                "Point A": [score_a, gap_a, comp_a, mag_a, pop_a],
                                "Point B": [score_b, gap_b, comp_b, mag_b, pop_b],
                            }
                        )
                        st.dataframe(comp_table, use_container_width=True)

                        st.subheader("Radar Chart (Strengths)")
                        _render_radar_chart(metrics_a, metrics_b, label_a="Point A", label_b="Point B")


if __name__ == "__main__":
    main()

