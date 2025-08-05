"""
season_from_geohash.py

Given a geohash (base‑32 string) anywhere in Australia and a
`datetime` (aware or naive), return a tuple (zone, season).

Zones  : 'temperate' or 'tropical'
Season : 'summer' | 'autumn' | 'winter' | 'spring'
"""

from datetime import datetime
import geohash  # pip install geohash  (or pygeohash)
import pandas as pd
import glob

# Constant: southern latitude of the Tropic of Capricorn
TROPIC_OF_CAPRICORN = -23.437  # decimal degrees

from datetime import datetime
from zoneinfo import ZoneInfo              # Py ≥3.9
# Robust geohash import (works with python‑geohash, geohash2 or pygeohash)
try:
    import geohash
except ModuleNotFoundError:
    try:
        import geohash2 as geohash
    except ModuleNotFoundError:
        import pygeohash as geohash

TROPIC_S = -23.437                        # Tropic of Capricorn latitude

# question	answer	answer_type	options	image_path	qa_id


question_one = "Which cropping season best matches the vegetation state in the image?"
chocies_one = ["winter_crop: crops that are planted in autumn and grow through the cool months", "summer_crop: crops that are planted in spring and grow through the warm months"]

question_two = "Which monsoonal season best matches the vegetation state in the image?"
chocies_two = ["wet_season: period with frequent rain and lush growth", "dry_season: period with little rain and drier fields"]

def prepare_vqa_template(season_info):
    if season_info == "wet_season":
        return {"question": question_two, "options": chocies_two, "answer": chocies_two[0]}
    elif season_info == "dry_season":
        return {"question": question_two, "options": chocies_two, "answer": chocies_two[1]}
    elif season_info == "summer_crop":
        return {"question": question_one, "options": chocies_one, "answer": chocies_one[1]}
    elif season_info == "winter_crop":
        return {"question": question_one, "options": chocies_one, "answer": chocies_one[0]}
    else:
        raise ValueError(f"Unknown season info: {season_info}")


def cropping_season(gh: str, date_str: str, four_labels=False):
    """Return agronomic season for an Australian cropland image."""
    lat, _ = geohash.decode(gh)
    month = datetime.strptime(date_str, "%Y-%m-%d").month

    if lat > TROPIC_S:            # Tropical North
        if 11 <= month or month <= 4:
            return "summer" if four_labels else "wet_season"
        else:
            return "winter" if four_labels else "dry_season"
    else:                         # Temperate grain belt
        if month in (12, 1, 2):
            return "summer" if four_labels else "summer_crop"
        if month in (3, 4, 5):
            return "autumn" if four_labels else "winter_crop"
        if month in (6, 7, 8):
            return "winter" if four_labels else "winter_crop"
        return "spring" if four_labels else "summer_crop"

# ----------------------------------------------------------------------
def _decode_lat_lon(hash_str: str) -> tuple[float, float]:
    """Decode a geohash to (lat, lon) using the lightweight `geohash` lib."""
    lat, lon = geohash.decode(hash_str)
    return lat, lon

# ----------------------------------------------------------------------
def _southern_hemisphere_season(dt: datetime) -> str:
    """Return meteorological season name for any Southern‑Hemisphere location."""
    month = dt.month
    if month in (12, 1, 2):
        return "summer"
    if month in (3, 4, 5):
        return "autumn"
    if month in (6, 7, 8):
        return "winter"
    return "spring"  # months 9,10,11

# ----------------------------------------------------------------------
def season_from_geohash(hash_str: str, dt: datetime | None = None) -> tuple[str, str]:
    """
    Parameters
    ----------
    hash_str : str
        Geohash covering any point in Australia.
    dt : datetime, optional
        The moment you want a season for (defaults to 'now' in system tz).

    Returns
    -------
    zone : str    ('temperate' | 'tropical')
    season : str  ('summer' | 'autumn' | 'winter' | 'spring')
    """
    if dt is None:
        dt = datetime.utcnow()

    lat, _ = _decode_lat_lon(hash_str)

    if lat > TROPIC_OF_CAPRICORN:   # remember: south latitudes are negative
        zone = "tropical"
    else:
        zone = "temperate"

    return zone, _southern_hemisphere_season(dt)