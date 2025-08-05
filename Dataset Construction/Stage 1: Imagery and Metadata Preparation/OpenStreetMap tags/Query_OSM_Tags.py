#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream-enriches a GeoJSON file with OpenStreetMap (OSM) tags using the
Overpass API and appends new features. A 7-character geohash of each
feature's centroid is used as a unique identifier to prevent duplicates
on re-runs.
"""

import json
import os
import sqlite3
import time

import fiona
import geohash  # pip install python-geohash
import geopandas as gpd
import overpy
from geopandas.io.file import infer_schema
from shapely.geometry import mapping
from tqdm import tqdm

# ======================================================================
# CONFIGURATION
# ======================================================================
INPUT_FILE = (
    "keep_4_landuse_metadata-categories.geojson"
)
OUTPUT_FILE = INPUT_FILE.replace(".geojson", "-landuse.geojson")
OVERPASS_SLEEP = 0.7  # Politeness delay between Overpass API calls (in seconds)

# ======================================================================
# 1. Set up SQLite Cache for Overpass Queries
# ======================================================================
db = sqlite3.connect("osm_query_cache.sqlite")
db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
db.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Logging for concurrency

# ======================================================================
# 2. Overpass API Helper Functions
# ======================================================================
api = overpy.Overpass()


def build_query(bbox):
    """Constructs an Overpass QL query string for a given bounding box."""
    minx, miny, maxx, maxy = bbox
    return f"""
    [out:json][timeout:25];
    (
      way["landuse"]({miny},{minx},{maxy},{maxx});
      way["natural"]({miny},{minx},{maxy},{maxx});
      way["water"]({miny},{minx},{maxy},{maxx});
      way["waterway"]({miny},{minx},{maxy},{maxx});
      way["leisure"]({miny},{minx},{maxy},{maxx});
      way["amenity"]({miny},{minx},{maxy},{maxx});
      way["highway"]({miny},{minx},{maxy},{maxx});
      way["man_made"]({miny},{minx},{maxy},{maxx});
      way["industrial"]({miny},{minx},{maxy},{maxx});
      way["place"]({miny},{minx},{maxy},{maxx});
    );
    out body;
    >;
    out skel qt;
    """


def fetch_osm_tags(bbox):
    """
    Fetches OSM tags for a bounding box, using a local cache to avoid
    redundant API calls.
    """
    key = "_".join(map(str, bbox))
    row = db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
    if row:
        return json.loads(row[0])

    try:
        result = api.query(build_query(bbox))
        tags = [way.tags for way in result.ways] + [rel.tags for rel in result.relations]
    except Exception as e:
        print(f"[WARNING] Overpass API error: {e}")
        tags = []

    # Cache the result
    db.execute(
        "INSERT OR REPLACE INTO cache(key, value) VALUES(?, ?)",
        (key, json.dumps(tags)),
    )
    db.commit()
    return tags


# ======================================================================
# 3. Prepare Input Data and Output Schema
# ======================================================================
print("[INFO] Reading input file...")
gdf_in = gpd.read_file(INPUT_FILE).to_crs(epsg=4326)

# Infer schema and add new properties
schema = infer_schema(gdf_in)
schema["properties"]["geometry_geohash"] = "str"
schema["properties"]["osm_keys"] = "str"

# ======================================================================
# 4. Resume Logic: Gather Geohashes from Existing Output File
# ======================================================================
seen_hashes = set()
append_mode = os.path.exists(OUTPUT_FILE)

if append_mode:
    print(f"[INFO] Found existing output file: {OUTPUT_FILE}")
    with fiona.open(OUTPUT_FILE, "r") as src:
        for feature in src:
            gh = feature["properties"].get("geometry_geohash")
            if gh:
                seen_hashes.add(gh)
    print(
        f"[INFO] Resuming with {len(seen_hashes):,} features already processed. "
        "New features will be appended."
    )

# ======================================================================
# 5. Process and Stream Features to Output File
# ======================================================================
open_mode = "a" if append_mode else "w"
with fiona.open(
    OUTPUT_FILE,
    open_mode,
    driver="GeoJSON",
    crs=gdf_in.crs.to_wkt(),
    schema=schema if not append_mode else None,  # Schema is only needed on creation
) as sink:
    print("[INFO] Starting feature enrichment...")
    for _, feature in tqdm(gdf_in.iterrows(), total=len(gdf_in), desc="Enriching"):
        geom = feature.geometry
        if not geom or not geom.is_valid:
            continue

        # Generate geohash and skip if already processed
        lat, lon = geom.centroid.y, geom.centroid.x
        gh = geohash.encode(lat, lon, precision=7)
        if gh in seen_hashes:
            continue
        seen_hashes.add(gh)

        # Fetch OSM tags
        time.sleep(OVERPASS_SLEEP)
        tags = fetch_osm_tags(geom.bounds)

        # Helper to extract unique values for a given key
        pull = lambda k: sorted({t[k] for t in tags if k in t})

        tag_dict = {
            "landuse": pull("landuse"),
            "natural": pull("natural"),
            "water": pull("water"),
            "waterway": pull("waterway"),
            "leisure": pull("leisure"),
            "amenity": pull("amenity"),
            "highway": pull("highway"),
            "man_made": pull("man_made"),
            "place": pull("place"),
            "industrial": pull("industrial"),
        }

        # Write the enriched feature to the output file
        sink.write(
            {
                "id": gh,
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    **feature.drop(labels="geometry").to_dict(),
                    "geometry_geohash": gh,
                    "osm_keys": json.dumps(tag_dict),
                },
            }
        )

print(f"\n[INFO] Finished. Output now contains {len(seen_hashes):,} unique features.")
db.close()
