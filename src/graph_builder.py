from __future__ import annotations

import glob
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Optional

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString


def _haversine(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """
    Great-circle distance between two (lon, lat) points in meters.
    """
    lon1, lat1 = p1
    lon2, lat2 = p2
    R = 6371000.0
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def _iter_line_coords(geom):
    if isinstance(geom, LineString):
        yield list(geom.coords)
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            yield list(line.coords)


def _pick_road_shp(shapefile_dir: Path) -> Path:
    candidates = glob.glob(str(shapefile_dir / "*roads*.shp"))
    if not candidates:
        raise FileNotFoundError(f"No road shapefile found under {shapefile_dir}")
    return Path(candidates[0])


def build_graph(shapefile_dir: str | Path, max_edges: int = 5000) -> nx.Graph:
    """
    Build a NetworkX graph from an OpenStreetMap roads shapefile.

    The shapefile directory is expected to contain a 'roads' shapefile.
    Only the first `max_edges` segments are used to keep the graph manageable.
    """
    shapefile_dir = Path(shapefile_dir)
    road_path = _pick_road_shp(shapefile_dir)

    # Read a slice to keep memory reasonable; we will still cap edges below.
    gdf = gpd.read_file(road_path, rows=slice(0, max_edges * 2))
    # Keep only line geometries and drop invalid ones
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
    if len(gdf) == 0:
        raise ValueError("Road shapefile contains no usable LineString geometry")

    if len(gdf) > max_edges:
        gdf = gdf.sample(n=max_edges, random_state=42)

    G = nx.Graph()

    edge_count = 0
    for _, row in gdf.iterrows():
        for coords in _iter_line_coords(row.geometry):
            if len(coords) < 2:
                continue
            for i in range(len(coords) - 1):
                p1 = (coords[i][0], coords[i][1])
                p2 = (coords[i + 1][0], coords[i + 1][1])
                dist = _haversine(p1, p2)
                G.add_node(p1, pos=p1)
                G.add_node(p2, pos=p2)
                G.add_edge(p1, p2, distance=dist, weight=dist, base_weight=dist)
                edge_count += 1
                if edge_count >= max_edges:
                    break
            if edge_count >= max_edges:
                break
        if edge_count >= max_edges:
            break

    if G.number_of_nodes() == 0:
        raise ValueError("Graph construction produced an empty graph")

    # Keep only the largest connected component to guarantee reachability
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    # Drop any isolates that may remain after subgraphing
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)

    return G


__all__ = ["build_graph"]
