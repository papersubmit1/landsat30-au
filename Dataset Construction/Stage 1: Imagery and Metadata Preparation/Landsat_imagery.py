import geopandas as gpd
from dea_tools.plotting import rgb
from datacube_ows.styles.api import apply_ows_style_cfg, xarray_image_as_png
from PIL import Image
import io
import datacube
import dea_tools.datahandling
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datacube.utils.geometry import Geometry
import xarray as xr
import pandas as pd
import os
import shutil
from dea_tools.datahandling import load_ard, xr_pansharpen        # DEA tools
from multiprocessing import Pool, cpu_count, set_start_method

# ------------------------------------------------------------------ #
#  Helper: percent NoData in a Dataset
# ------------------------------------------------------------------ #
def num_pct_valid(ds: xr.Dataset, band: str = "fmask") -> float:
    # 1️⃣ build a boolean mask of valid pixels
    # NaN values in ds[band] will result in False in valid_mask, effectively ignoring them.
    valid_mask = ds[band].isin([1, 4, 5, -999])


    # 2️⃣ count along the spatial dims (y, x)
    #    this gives a DataArray indexed by time
    n_valid = valid_mask.sum(dim=("y", "x"))

    # if you also want the *total* number of pixels (per time):
    # .notnull() ensures we only count pixels that are NOT NaN,
    # so the total reflects actual data points.
    n_total = ds[band].notnull().sum(dim=("y", "x"))

    # 3️⃣ (optional) compute percentage valid
    pct_valid = (n_valid / n_total) * 100

    return pct_valid.item()

def get_time_range_from_year(year) -> tuple[str, str]:

    year = int(year)
    
    quarter_number = year % 4

    start_date = None
    end_date = None

    if quarter_number == 0:
        # Case 0: Full year (Jan 1 to Dec 31 of the input year)
        # Example: 2000 -> (2000-01-01, 2000-12-31)
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
    elif quarter_number == 1:
        start_date = f"{year}-04-01"
        end_date = f"{year+1}-03-31"
    elif quarter_number == 2:
        start_date = f"{year}-07-01"
        end_date = f"{year+1}-06-30"
    elif quarter_number == 3:
        start_date = f"{year}-10-01"
        end_date = f"{year+1}-09-30"
    else:
        # This case should theoretically not be reached with % 4, but included for robustness
        print(f"Unexpected quarter number: {quarter_number}")
        return None, None

    return (start_date, end_date) # Return only the date part


def zip_folder(folder_path: str, output_zip_path: str) -> None:
    """Zip an entire folder (shutil.make_archive wrapper)."""
    shutil.make_archive(output_zip_path, "zip", folder_path)
    print(f"✓ Zipped '{folder_path}' ➜ '{output_zip_path}.zip'")


# ------------------------------------------------------------------ #
# 2. Worker: one region/year per process
# ------------------------------------------------------------------ #
def process_region(task):
    (
        year, region_code, region_df,                     # task specifics
        PRODUCT, PATCH_SIZE, PAN_RES, RGB_CFG,            # global constants
        RAW_SFX, PS_SFX, clean_thresh,
    ) = task

    # Datacube must be opened *inside* the subprocess
    dc = datacube.Datacube(app=f"Export_{region_code}")

    out_dir = Path(f"DEA_VLM_images/{PRODUCT}-{region_code}-{year}-patches")
    meta_csv = out_dir.parent / f"{out_dir.name}_metadata.csv"

    if meta_csv.exists():
        print(f"[{region_code}] already done; skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    image_urls, geohashes, timestamps = [], [], []

    # --- iterate geohashes in this region -----------------------------------
    for geo_hash in region_df.geometry_geohash:
        geom = region_df.loc[
            region_df.geometry_geohash == geo_hash, "geometry"
        ].iloc[0]
        gp = Geometry(geom, crs=region_df.crs)

        time_range = get_time_range_from_year(year)

        # 1️⃣ fmask to check cloud cover
        ds_fmask = load_ard(
            dc=dc,
            products=[PRODUCT],
            # measurements=["fmask", "nbart_red"],
            measurements=["fmask"],
            geopolygon=gp,
            time=time_range,
            output_crs="EPSG:3577",
            resolution=(-30, 30),
            resampling="bilinear",
            dataset_maturity="final",
            # group_by="solar_day",
            dask_chunks={"time": 1, "y": PATCH_SIZE, "x": PATCH_SIZE},
        )

        for t in range(ds_fmask.dims["time"]):
            try:
                if num_pct_valid(ds_fmask.isel(time=t).load()) <= clean_thresh:
                    continue  # too cloudy – next timestep
                # 2️⃣ RAW RGB @30 m
                ds_raw = load_ard(
                    dc=dc,
                    products=[PRODUCT],
                    measurements=["nbart_red", "nbart_green", "nbart_blue"],
                    geopolygon=gp,
                    time=time_range,
                    mask_pixel_quality=False,
                    output_crs="EPSG:3577",
                    resolution=(-30, 30),
                    resampling="bilinear",
                    dataset_maturity="final",
                    dask_chunks={"time": 1, "y": PATCH_SIZE, "x": PATCH_SIZE},
                )
    
                raw_slice = ds_raw.isel(time=t).load()
    
                #pan_slice = ds_pan.isel(time=t).load()        # 15 m slice
    
                #ps_slice = xr_pansharpen(
                #    pan_slice[["nbart_red", "nbart_green", "nbart_blue", "nbart_panchromatic"]],
                #    transform="brovey", pan_band="nbart_panchromatic"
                #)
    
                date_str = str(raw_slice.time.values).split("T")[0]
                raw_fn = f"{PRODUCT}-{region_code}-{year}-{geo_hash}-{date_str}-{RAW_SFX}"
                #ps_fn  = f"{PRODUCT}-{region_code}-{year}-{geo_hash}-{date_str}-{PS_SFX}"
    
                # save PNGs
                Image.open(io.BytesIO(xarray_image_as_png(apply_ows_style_cfg(RGB_CFG, raw_slice)))).save(out_dir / raw_fn)
                #Image.open(io.BytesIO(xarray_image_as_png(apply_ows_style_cfg(RGB_CFG, ps_slice )))).save(out_dir / ps_fn)
    
                image_urls.append(f"{out_dir}/{raw_fn}")
                geohashes.append(geo_hash)
                timestamps.append(date_str)
    
                print(f"[{region_code}] ✓ {raw_fn} ")
                break  # first clean timestep only
            except Exception as e:
              print(f"An exception occurred {e}, cannot load curent ARD, skipt to next one")
    print(f"[{region_code}] done → {meta_csv}")


# ------------------------------------------------------------------ #
# 3. Driver
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Safer for Windows / notebooks
    set_start_method("spawn", force=True)

    # -------- load GeoJSON + add region_code --------
    # gdf = gpd.read_file("fine_tune_files.geojson")
    gdf = gpd.read_file("keep_4_landuse_metadata-categories.geojson")
    
    gdf["region_code"] = [
        s.split("-2020")[0].replace("ga_ls8cls9c_gm_cyear_3-", "")
        for s in gdf["patch_id"]
    ]

    # ----------- user‑tunable constants -------------
    #PRODUCT        = "ga_ls9c_ard_3"
    #YEARS          = ["2023", "2022"]          

    PRODUCT        = "ga_ls5t_ard_3"
    YEARS          = ["2004", 
                      "2005",
                      "2006",
                      "2007",
                      "2008",
                      "2009",
                      "2010",
                      "1989",
                      "1990",
                      "1991",
                      "1992",
                      "1993",
                      "1994",
                      "1995",
                      "1996",
                      "1997",
                      "1998",
                      ]
    
    #PRODUCT        = "ga_ls7e_ard_3"
    #YEARS          = ["2001", 
    #                  "2015", 
    #                  "2002", 
    #                  ]         
    
    PATCH_SIZE     = 256               # pixels for Dask chunks
    PAN_RES        = 15                # metres
    RAW_SFX, PS_SFX = "raw.png", "ps.png"
    clean_pixel_threshold = 99.5       # % cloud‑free required

    RGB_CFG = {
        "components": {
            "red":   {"nbart_red":   1.0},
            "green": {"nbart_green": 1.0},
            "blue":  {"nbart_blue":  1.0},
        },
        "scale_range": (0, 3000),
    }

    # ----------- build task list -------------
    tasks = []
    for year in YEARS:
        for rc in sorted(set(gdf["region_code"])):
            tasks.append((
                year, rc, gdf[gdf["region_code"] == rc].copy(),
                PRODUCT, PATCH_SIZE, PAN_RES, RGB_CFG,
                RAW_SFX, PS_SFX, clean_pixel_threshold,
            ))

    print(f"Launching {len(tasks)} jobs across up to {cpu_count()} cores …")

    # ----------- run the pool ---------------
    with Pool(processes=min(8, len(tasks))) as pool:
        for _ in tqdm(pool.imap_unordered(process_region, tasks), total=len(tasks)):
            pass

    
    year_info = "-".join([str(e) for e in YEARS])
    
    # ----------- optional: zip everything ----
    zip_folder("DEA_VLM_images", f"DEA_VLM_images_{PRODUCT}_ARD_{str(clean_pixel_threshold)}")

    import glob
    all_csv_files = glob.glob("DEA_VLM_images/*.csv")
    all_dfs = []
    
    import pandas as pd
    
    for csv_file in all_csv_files:
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
    
    all_df = pd.concat(all_dfs, ignore_index=True)

    
    all_df.to_csv(f"DEA_VLM_images_{PRODUCT}_ARD_{str(clean_pixel_threshold)}.csv", index=False)