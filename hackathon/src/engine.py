from __future__ import annotations

import geopandas as gpd
import h3
from shapely.geometry import Polygon


def _h3_hex_to_polygon(h3_index: str) -> Polygon:
    # h3.cell_to_boundary returns an iterable of (lat, lng) pairs.
    # shapely expects (lng, lat) coordinate order.
    boundary = h3.cell_to_boundary(h3_index)
    coords = [(lng, lat) for lat, lng in boundary]
    return Polygon(coords)


def generate_hex_grid(
    lat: float,
    lon: float,
    radius_km: float,
    resolution: int,
) -> gpd.GeoDataFrame:
    """
    Generate an H3 hex grid around a center point.

    The grid is created by taking the center hex and expanding outward based
    on estimated hex edge distance at the chosen resolution.
    """

    if radius_km <= 0:
        raise ValueError("radius_km must be > 0")
    if resolution < 0:
        raise ValueError("resolution must be >= 0")

    center_h3 = h3.latlng_to_cell(lat, lon, resolution)

    # Convert km to k-ring distance approximately.
    # This is a heuristic; for a hackathon scaffold it's fine.
    avg_hex_edge_km = max(h3.average_hexagon_edge_length(resolution, unit="km"), 0.001)
    k_ring = int(radius_km / avg_hex_edge_km) + 1

    hexes = list(h3.grid_disk(center_h3, k_ring))

    gdf = gpd.GeoDataFrame(
        {"h3_index": hexes},
        geometry=[_h3_hex_to_polygon(h) for h in hexes],
        crs="EPSG:4326",
    )

    return gdf


def _ensure_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure geometries are points (for spatial joins / counts).
    """
    geom_type = gdf.geometry.geom_type
    if geom_type.iloc[0] == "Point":
        return gdf

    # For lines/polygons, use representative points for a stable centroid-like location.
    return gdf.copy().assign(geometry=gdf.geometry.representative_point())


def _count_points_in_hexes(
    hex_grid: gpd.GeoDataFrame,
    features: gpd.GeoDataFrame,
    count_col: str,
) -> gpd.GeoDataFrame:
    """
    Count features falling within each H3 hex polygon.
    """
    if features.empty:
        return hex_grid.assign(**{count_col: 0.0})

    if features.crs is None:
        features = features.set_crs("EPSG:4326")
    else:
        features = features.to_crs("EPSG:4326")

    features_points = _ensure_points(features)

    # Spatial join counts features in polygons.
    joined = gpd.sjoin(features_points, hex_grid[["h3_index", "geometry"]], how="inner", predicate="within")
    counts = joined.groupby("h3_index").size().rename(count_col).reset_index()

    out = hex_grid.merge(counts, on="h3_index", how="left")
    out[count_col] = out[count_col].fillna(0).astype(float)
    return out


def _add_population_proxy(
    hex_grid: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    column_name: str = "population_proxy",
) -> gpd.GeoDataFrame:
    """
    Estimate a residential density proxy per hexagon based on building floor area.

    Logic:
    - For each building:
      - Take geometry area (in projected meters).
      - Determine levels from `building:levels` if present, else assume 2.
      - floor_area = area * levels.
    - Assign each building to a hex by representative point (centroid-like).
    - Sum floor_area per hex -> population_proxy.
    """
    if buildings.empty:
        return hex_grid.assign(**{column_name: 0.0})

    # Work in a projected CRS for area calculations.
    if hex_grid.crs is None:
        hex_grid = hex_grid.set_crs("EPSG:4326")
    if buildings.crs is None:
        buildings = buildings.set_crs("EPSG:4326")

    hex_proj = hex_grid.to_crs("EPSG:3857")
    bld_proj = buildings.to_crs("EPSG:3857")

    # Compute building area and levels.
    areas = bld_proj.geometry.area.astype(float)
    # `building:levels` may be stored as string; parse to float where possible, else fallback to 2.
    levels_raw = bld_proj.get("building:levels")
    if levels_raw is None:
        levels = 2.0
        levels_series = None
    else:
        levels_series = []
        for val in levels_raw:
            try:
                levels_series.append(float(val))
            except (TypeError, ValueError):
                levels_series.append(2.0)

    if levels_series is None:
        floor_area = areas * 2.0
    else:
        import pandas as _pd  # local import to avoid top-level dependency

        levels_series = _pd.Series(levels_series, index=bld_proj.index).astype(float)
        floor_area = areas * levels_series

    # Attach floor_area as a column and collapse to representative points.
    bld_proj = bld_proj.copy()
    bld_proj["floor_area"] = floor_area
    bld_points = _ensure_points(bld_proj)

    # Spatial join: assign each building to a hex.
    joined = gpd.sjoin(
        bld_points[["floor_area", "geometry"]],
        hex_proj[["h3_index", "geometry"]],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        return hex_grid.assign(**{column_name: 0.0})

    grouped = joined.groupby("h3_index")["floor_area"].sum().rename(column_name).reset_index()
    hex_with_pop = hex_grid.merge(grouped, on="h3_index", how="left")
    hex_with_pop[column_name] = hex_with_pop[column_name].fillna(0.0).astype(float)
    return hex_with_pop


def _add_competitive_gap(
    hex_grid: gpd.GeoDataFrame,
    competitor_features: gpd.GeoDataFrame,
    *,
    radius_km: float,
    k_nearest: int = 3,
) -> gpd.GeoDataFrame:
    """
    For each hexagon, compute distance (meters) to the k nearest competitors.

    Outputs:
    - nearest_competitor_1m..k m
    - competitor_gap_m: mean distance to k nearest (or a large value if none exist)
    - gap_score: 0-10, higher when competitors are farther (under-served)
    """
    if hex_grid.empty:
        return hex_grid

    if competitor_features.empty:
        out = hex_grid.copy()
        for i in range(1, k_nearest + 1):
            out[f"nearest_competitor_{i}m"] = float(radius_km) * 2000.0
        out["competitor_gap_m"] = float(radius_km) * 2000.0
        out["gap_score"] = 10.0
        return out

    if hex_grid.crs is None:
        hex_grid = hex_grid.set_crs("EPSG:4326")
    if competitor_features.crs is None:
        competitor_features = competitor_features.set_crs("EPSG:4326")

    # Project for distance computations.
    hex_proj = hex_grid.to_crs("EPSG:3857").copy()
    comp_pts = _ensure_points(competitor_features).to_crs("EPSG:3857")

    # Hex centroid points.
    centroids = hex_proj.geometry.centroid

    import numpy as _np  # local import

    comp_xy = _np.array([(g.x, g.y) for g in comp_pts.geometry if g is not None])
    if comp_xy.size == 0:
        out = hex_grid.copy()
        for i in range(1, k_nearest + 1):
            out[f"nearest_competitor_{i}m"] = float(radius_km) * 2000.0
        out["competitor_gap_m"] = float(radius_km) * 2000.0
        out["gap_score"] = 10.0
        return out

    # Compute nearest distances for each centroid (brute force).
    nearest_cols: dict[str, list[float]] = {f"nearest_competitor_{i}m": [] for i in range(1, k_nearest + 1)}
    gap_vals: list[float] = []

    fallback_far = float(radius_km) * 2000.0  # 2x radius in meters

    for c in centroids:
        if c is None:
            dists = _np.array([fallback_far] * k_nearest, dtype=float)
        else:
            dx = comp_xy[:, 0] - float(c.x)
            dy = comp_xy[:, 1] - float(c.y)
            all_d = _np.sqrt(dx * dx + dy * dy)
            if all_d.size == 0:
                dists = _np.array([fallback_far] * k_nearest, dtype=float)
            else:
                all_d.sort()
                # pad if fewer than k competitors
                if all_d.size < k_nearest:
                    pad = _np.full((k_nearest - all_d.size,), fallback_far, dtype=float)
                    dists = _np.concatenate([all_d, pad])
                else:
                    dists = all_d[:k_nearest]

        for i in range(1, k_nearest + 1):
            nearest_cols[f"nearest_competitor_{i}m"].append(float(dists[i - 1]))
        gap_vals.append(float(_np.mean(dists)))

    out = hex_grid.copy()
    for k, v in nearest_cols.items():
        out[k] = _np.array(v, dtype=float)
    out["competitor_gap_m"] = _np.array(gap_vals, dtype=float)

    # Normalize to 0-10, higher is better (farther competitors).
    max_gap = float(out["competitor_gap_m"].max()) if not out["competitor_gap_m"].empty else 0.0
    if max_gap > 0:
        out["gap_score"] = (out["competitor_gap_m"] / max_gap) * 10.0
    else:
        out["gap_score"] = 0.0

    return out


def find_optimal_locations(
    lat: float,
    lon: float,
    radius_km: float,
    resolution: int,
    competitor_features: gpd.GeoDataFrame,
    magnet_features: gpd.GeoDataFrame,
    residential_buildings: gpd.GeoDataFrame | None = None,
    focus_under_served: bool = False,
    top_k: int | None = 5,
) -> gpd.GeoDataFrame:
    """
    Generate hexes, score them using weighted competitor/magnet counts, and return top-k.
    """
    hex_grid = generate_hex_grid(lat=lat, lon=lon, radius_km=radius_km, resolution=resolution)

    hexes_with_competitors = _count_points_in_hexes(
        hex_grid=hex_grid,
        features=competitor_features,
        count_col="competitor_count",
    )
    hexes_with_magnets = _count_points_in_hexes(
        hex_grid=hex_grid,
        features=magnet_features,
        count_col="magnet_count",
    )

    # Merge magnet counts by `h3_index` only (avoids geometry equality issues).
    scored = hexes_with_competitors.merge(
        hexes_with_magnets[["h3_index", "magnet_count"]],
        on="h3_index",
        how="left",
    )

    # Add residential density proxy if buildings are provided.
    if residential_buildings is not None:
        scored = _add_population_proxy(scored, residential_buildings, column_name="population_proxy")
    else:
        scored = scored.assign(population_proxy=0.0)

    # Competitive gap (under-served-ness).
    scored = _add_competitive_gap(scored, competitor_features, radius_km=radius_km, k_nearest=3)

    scored["competitor_count"] = scored["competitor_count"].fillna(0.0).astype(float)
    scored["magnet_count"] = scored["magnet_count"].fillna(0.0).astype(float)
    scored["population_proxy"] = scored["population_proxy"].fillna(0.0).astype(float)
    scored["gap_score"] = scored.get("gap_score", 0.0)
    scored["gap_score"] = scored["gap_score"].fillna(0.0).astype(float)

    # Raw weighted score:
    # - Magnets increase score
    # - Competitors decrease score
    # - More residential floor area (population_proxy) increases score
    scored["raw_score"] = (scored["magnet_count"] * 2.0) - (scored["competitor_count"] * 1.2)

    max_pop = float(scored["population_proxy"].max()) if not scored["population_proxy"].empty else 0.0
    if max_pop > 0:
        # Normalize population proxy into [0, 1] and weight it.
        scored["pop_component"] = (scored["population_proxy"] / max_pop) * 3.0
        scored["raw_score"] = scored["raw_score"] + scored["pop_component"]
    else:
        scored["pop_component"] = 0.0

    # Gap component: reward hexes farther from competitors.
    gap_weight = 2.0 if focus_under_served else 0.6
    scored["gap_component"] = (scored["gap_score"] / 10.0) * gap_weight
    scored["raw_score"] = scored["raw_score"] + scored["gap_component"]

    # Normalize to 0-10 scale.
    max_raw = float(scored["raw_score"].max()) if not scored["raw_score"].empty else 0.0
    if max_raw > 0:
        scored["score"] = (scored["raw_score"].clip(lower=0.0) / max_raw) * 10.0
    else:
        scored["score"] = 0.0

    scored = scored.sort_values("score", ascending=False)
    if top_k is None or top_k <= 0:
        return scored
    return scored.head(top_k)

