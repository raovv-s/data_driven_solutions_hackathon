from __future__ import annotations

from typing import Any, Iterable, Tuple

import networkx as nx

import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point


def load_osm_features(
    lat: float,
    lon: float,
    distance_m: float,
    tags: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    """
    Load OSM features around a point.

    Parameters
    ----------
    lat, lon:
        Center point in WGS84.
    distance_m:
        Search radius in meters.
    tags:
        OSM tag filter passed to OSMnx (e.g. {"shop": True} or {"amenity": ["restaurant", "cafe"]}).
    """

    if tags is None:
        tags = {"amenity": True}

    # OSMnx uses a lat/lon point geometry internally.
    point = (lat, lon)

    # OSMnx returns GeoDataFrames.
    # osmnx>=2.x uses `center_point=` as the keyword argument.
    try:
        gdf = ox.features_from_point(center_point=point, tags=tags, dist=distance_m)
    except Exception:
        # Happens when there are no matching OSM features for the query/tags,
        # or when Overpass/network requests fail.
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if gdf.empty:
        return gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    # Ensure consistent CRS.
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def generate_walk_isochrones(
    lat: float,
    lon: float,
    walk_minutes: Iterable[int] = (5, 10),
    *,
    network_type: str = "walk",
    # Distance used to download the network around the center.
    # Roughly scales with walking time.
    dist_m: int = 2500,
    # Constant average walking speed used to convert edge length -> travel_time.
    walk_speed_kph: float = 4.8,
    # Buffer around reachable nodes when building the final polygon.
    buffer_m: float = 60.0,
) -> gpd.GeoDataFrame:
    """
    Generate walking isochrone polygons for the given minutes.

    This uses:
    - osmnx.graph_from_point(..., network_type="walk")
    - osmnx.project_graph
    - Dijkstra shortest paths (cutoff by travel time)
    """
    try:
        G = ox.graph_from_point((lat, lon), dist=dist_m, network_type=network_type)
    except Exception:
        # If the Overpass query fails or returns nothing, don't crash the app.
        return gpd.GeoDataFrame({"walk_minutes": [], "geometry": []}, crs="EPSG:4326")

    if len(G) == 0:
        return gpd.GeoDataFrame({"walk_minutes": [], "geometry": []}, crs="EPSG:4326")

    G_proj = ox.project_graph(G)

    # Convert center point to the projected CRS of the network.
    center_pt = Point(lon, lat)
    proj_crs = G_proj.graph.get("crs")
    if proj_crs is None:
        return gpd.GeoDataFrame({"walk_minutes": [], "geometry": []}, crs="EPSG:4326")
    proj_pt, _ = ox.projection.project_geometry(center_pt, to_crs=proj_crs)
    # osmnx's nearest_nodes requires scipy (optional dependency).
    # If scipy isn't installed, fall back to a simple "min distance" scan.
    try:
        origin_node = ox.distance.nearest_nodes(G_proj, X=proj_pt.x, Y=proj_pt.y)
    except ImportError:
        px, py = float(proj_pt.x), float(proj_pt.y)
        origin_node = min(
            G_proj.nodes(data=True),
            key=lambda n: (float(n[1].get("x", 0.0)) - px) ** 2 + (float(n[1].get("y", 0.0)) - py) ** 2,
        )[0]

    speed_mps = (walk_speed_kph * 1000.0) / 3600.0
    # Define travel_time on each edge (seconds).
    for u, v, k, data in G_proj.edges(keys=True, data=True):
        length_m = data.get("length")
        if length_m is None:
            continue
        data["travel_time"] = float(length_m) / speed_mps

    polygons: list[dict[str, Any]] = []

    for minutes in walk_minutes:
        cutoff_s = float(minutes) * 60.0
        try:
            reachable = nx.single_source_dijkstra_path_length(
                G_proj,
                origin_node,
                cutoff=cutoff_s,
                weight="travel_time",
            )
        except Exception:
            reachable = {}

        node_ids = list(reachable.keys())
        if not node_ids:
            polygons.append({"walk_minutes": minutes, "geometry": None})
            continue

        # Build an approximate isochrone polygon:
        # buffer reachable nodes, union them, then take the convex hull.
        pts = [Point(G_proj.nodes[n]["x"], G_proj.nodes[n]["y"]) for n in node_ids]
        nodes_gdf = gpd.GeoDataFrame(geometry=pts, crs=G_proj.graph.get("crs"))
        buffered = nodes_gdf.buffer(buffer_m)
        unioned = buffered.unary_union
        poly_proj = unioned.convex_hull

        poly_4326 = gpd.GeoSeries([poly_proj], crs=G_proj.graph.get("crs")).to_crs("EPSG:4326").iloc[0]
        polygons.append({"walk_minutes": minutes, "geometry": poly_4326})

    # Keep only valid polygons; if any are None, they will be dropped.
    out = gpd.GeoDataFrame(polygons, crs="EPSG:4326")
    out = out[out["geometry"].notna()].reset_index(drop=True)
    return out


def get_building_data(
    center_point: Tuple[float, float],
    radius_m: float,
) -> gpd.GeoDataFrame:
    """
    Fetch all buildings around a center point from OSM.

    center_point: (lat, lon)
    radius_m: search radius in meters
    """
    lat, lon = center_point
    # Pull all buildings; OSMnx returns attributes like "building:levels" when present.
    return load_osm_features(
        lat=lat,
        lon=lon,
        distance_m=radius_m,
        tags={"building": True},
    )


def simulate_competitor_ratings(competitors: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Simulate competitor ratings (e.g., Google/Yelp/2GIS) in a deterministic way.

    Adds a `rating` column in the range [2.6, 4.9]. This is a stand-in for a real
    review API or scraper.
    """
    if competitors.empty:
        out = competitors.copy()
        out["rating"] = []
        return out

    out = competitors.copy()

    def _stable_rating(row) -> float:
        # Prefer OSM identifiers if available; otherwise fall back to geometry.
        base = None
        for key in ("osmid", "osm_id", "id"):
            if key in row and row[key] is not None:
                base = str(row[key])
                break
        if base is None:
            g = row.get("geometry")
            if g is None:
                base = "0"
            else:
                base = f"{getattr(g, 'x', 0.0):.5f},{getattr(g, 'y', 0.0):.5f}"

        import hashlib

        h = int(hashlib.sha256(base.encode("utf-8")).hexdigest()[:8], 16) % 1000
        # Map hash bucket to rating range.
        return 2.6 + (h / 1000.0) * (4.9 - 2.6)

    out["rating"] = out.apply(_stable_rating, axis=1).astype(float)
    return out

