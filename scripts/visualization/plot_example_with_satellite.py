"""
Generate trajectory plots + before/after satellite imagery for example pixels
used in recovery_assessment_guide.md.

Each pixel gets:
  1. A 1x2 trajectory plot (GPP | SVH) with natural/transformed/KNN envelopes
  2. A 1x2 before/after satellite image (Sentinel-2 true color, ~2016 vs ~2022)

Supports pixels from any ecoregion by loading ecoregion-specific natural reference.

Usage:
    python scripts/visualization/plot_example_with_satellite.py
"""
import sys
import os
import ee
import io
import time
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from PIL import Image
from sklearn.neighbors import BallTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
DATA_DIR = BASE_DIR / "data"
ABANDONED_AG = DATA_DIR / "abandoned_ag_gpp_2000_2022_SA.parquet"
INDICES_GPP = DATA_DIR / "indices_gpp_svh_2000_2022.parquet"
NAT_SUBSET = DATA_DIR / "dfsubsetNatural.parquet"
OUT_DIR = BASE_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

YEARS = list(range(2000, 2023))
GPP_COLS = [f"GPP_{y}" for y in YEARS]
SVH_COLS = [f"SVH_{y}" for y in YEARS]
SVH_SCALE = 0.1  # raw SVH values must be multiplied by 0.1
KNN_K = 10
KNN_RADIUS_KM = 50.0
EARTH_R_KM = 6371.0

# Example pixels: label, eco_id, pixel_id, lat, lon, description
# lat/lon are authoritative (from score files)
PIXELS = [
    {
        "label": "A", "eco_id": 81, "pixel_id": 8393,
        "desc": "Moderate recovery (score 53.6)",
    },
    {
        "label": "B", "eco_id": 81, "pixel_id": 34240,
        "desc": "Advanced recovery (score 82.2)",
    },
    {
        "label": "C", "eco_id": 81, "pixel_id": 79489,
        "desc": "Early/limited recovery (score 24.1)",
    },
    {
        "label": "D", "eco_id": 38, "pixel_id": 80892,
        "lat": -25.2828, "lon": 26.4084,
        "desc": "High compositional convergence (score 94.1)",
    },
]

GEE_PROJECT = "ee-gsingh"
SAT_BUFFER_M = 500  # buffer around pixel for satellite thumbnail
SAT_SCALE = 5       # meters per pixel in thumbnail


def init_gee():
    try:
        ee.Initialize(project=GEE_PROJECT,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    print("[OK] GEE initialized")


def retry_gee(fn, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
                print(f"  Retry {attempt+1}: {e}")
            else:
                raise


def load_natural_reference(eco_id):
    """Load natural reference with GPP/SVH time series for given ecoregion."""
    gpp_sel = ", ".join(GPP_COLS)
    svh_sel = ", ".join(SVH_COLS)

    if eco_id == 81:
        # Use sanlc_2022 IN (1,2) as proxy for natural — avoids slow JOIN
        sql = f"""
            SELECT latitude, longitude, {gpp_sel}, {svh_sel}
            FROM '{INDICES_GPP}'
            WHERE sanlc_2022 IN (1, 2) AND eco_id = {eco_id}
            USING SAMPLE reservoir(10000 ROWS) REPEATABLE(42)
        """
    else:
        ref_path = DATA_DIR / f"ref_samples_eco{eco_id}.parquet"
        sql = f"""
            SELECT latitude, longitude, {gpp_sel}, {svh_sel}
            FROM '{ref_path}'
            WHERE "natural" = 1
        """
    conn = duckdb.connect()
    conn.execute("SET memory_limit='2GB'")
    df = conn.sql(sql).df()
    conn.close()
    print(f"  Natural reference: {len(df):,} pixels (eco_id={eco_id})")
    return df


def load_transformed_reference(eco_id, n=5000):
    """Load transformed reference for the ecoregion."""
    gpp_sel = ", ".join(GPP_COLS)
    svh_sel = ", ".join(SVH_COLS)

    if eco_id == 81:
        sql = f"""
            SELECT latitude, longitude, {gpp_sel}, {svh_sel}
            FROM '{INDICES_GPP}'
            WHERE sanlc_2022 NOT IN (1, 2) AND eco_id = {eco_id}
            USING SAMPLE reservoir({n} ROWS) REPEATABLE(42)
        """
    else:
        ref_path = DATA_DIR / f"ref_samples_eco{eco_id}.parquet"
        sql = f"""
            SELECT latitude, longitude, {gpp_sel}, {svh_sel}
            FROM '{ref_path}'
            WHERE "natural" = 0
            USING SAMPLE reservoir({min(n, 50000)} ROWS) REPEATABLE(42)
        """
    conn = duckdb.connect()
    conn.execute("SET memory_limit='2GB'")
    df = conn.sql(sql).df()
    conn.close()
    print(f"  Transformed reference: {len(df):,} pixels")
    return df


def load_test_pixel(pixel_id, eco_id, lat=None, lon=None):
    """Load test pixel time series, matching by lat/lon if provided."""
    gpp_sel = ", ".join(GPP_COLS)
    svh_sel = ", ".join(SVH_COLS)

    if lat is not None and lon is not None:
        sql = f"""
            SELECT pixel_id, latitude, longitude, eco_id, sanlc_2022,
                   {gpp_sel}, {svh_sel}
            FROM '{ABANDONED_AG}'
            WHERE abs(latitude - ({lat})) < 0.001
              AND abs(longitude - ({lon})) < 0.001
              AND eco_id = {eco_id}
            LIMIT 1
        """
    else:
        sql = f"""
            SELECT pixel_id, latitude, longitude, eco_id, sanlc_2022,
                   {gpp_sel}, {svh_sel}
            FROM '{ABANDONED_AG}'
            WHERE pixel_id = {pixel_id} AND eco_id = {eco_id}
            LIMIT 1
        """
    conn = duckdb.connect()
    conn.execute("SET memory_limit='2GB'")
    df = conn.sql(sql).df()
    conn.close()
    if len(df) == 0:
        raise RuntimeError(
            f"Pixel {pixel_id} (eco_id={eco_id}) not found in abandoned_ag")
    return df.iloc[0]


def find_knn(test_lat, test_lon, nat_df, k=KNN_K):
    """Find K nearest natural neighbours."""
    coords_rad = np.radians(nat_df[["latitude", "longitude"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    test_rad = np.radians([[test_lat, test_lon]])
    _, idx = tree.query(test_rad, k=min(k, len(nat_df)))
    return nat_df.iloc[idx[0]]


def get_satellite_thumbnail(lat, lon, year_start, year_end, buffer_m=SAT_BUFFER_M):
    """Get Sentinel-2 true color thumbnail from GEE."""
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(buffer_m).bounds()

    # Use Sentinel-2 SR harmonised
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(roi)
          .filterDate(f"{year_start}-01-01", f"{year_end}-12-31")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .select(["B4", "B3", "B2"])
          .median()
          .clip(roi))

    # Visualisation parameters for true color
    vis = {"min": 0, "max": 3000, "bands": ["B4", "B3", "B2"]}

    url = retry_gee(lambda: s2.getThumbURL({
        "region": roi,
        "dimensions": "256x256",
        "format": "png",
        **vis,
    }))

    import urllib.request
    response = retry_gee(lambda: urllib.request.urlopen(url, timeout=30))
    img_data = response.read()
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)


def plot_trajectory(test, nat_df, trans_df, knn_df, label, desc, out_path):
    """Create a 1x2 trajectory plot for a single pixel."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Pixel {label}: {desc}\n"
        f"pixel_id={test['pixel_id']}  |  "
        f"lat={test['latitude']:.3f}, lon={test['longitude']:.3f}  |  "
        f"eco_id={int(test['eco_id'])}  |  SANLC={int(test['sanlc_2022'])}",
        fontsize=11, fontweight="bold",
    )

    # Drop rows with NaN before converting to float
    nat_clean = nat_df[GPP_COLS + SVH_COLS].dropna()
    trans_clean = trans_df[GPP_COLS + SVH_COLS].dropna()
    nat_gpp = nat_clean[GPP_COLS].values.astype(float)
    nat_svh = nat_clean[SVH_COLS].values.astype(float) * SVH_SCALE
    trans_gpp = trans_clean[GPP_COLS].values.astype(float)
    trans_svh = trans_clean[SVH_COLS].values.astype(float) * SVH_SCALE
    knn_gpp_mean = knn_df[GPP_COLS].values.astype(float).mean(axis=0)
    knn_svh_mean = knn_df[SVH_COLS].values.astype(float).mean(axis=0) * SVH_SCALE
    test_gpp = test[GPP_COLS].values.astype(float)
    test_svh = test[SVH_COLS].values.astype(float) * SVH_SCALE

    for ax, var, nat_vals, trans_vals, knn_mean, test_vals, ylabel in [
        (axes[0], "GPP", nat_gpp, trans_gpp, knn_gpp_mean, test_gpp,
         "GPP (gC/m\u00b2/yr)"),
        (axes[1], "SVH", nat_svh, trans_svh, knn_svh_mean, test_svh,
         "SVH (m)"),
    ]:
        nat_mean = nat_vals.mean(axis=0)
        nat_std = nat_vals.std(axis=0)
        ax.fill_between(YEARS, nat_mean - nat_std, nat_mean + nat_std,
                         alpha=0.2, color="#2ecc71", label="Natural \u00b11 SD")
        ax.plot(YEARS, nat_mean, "--", color="#2ecc71", lw=1.5,
                label="Natural mean")
        ax.plot(YEARS, trans_vals.mean(axis=0), "--", color="#e67e22", lw=1.5,
                label="Transformed mean")
        ax.plot(YEARS, knn_mean, "--", color="#3498db", lw=1.5,
                label=f"KNN mean (K={len(knn_df)})")
        ax.plot(YEARS, test_vals, "-", color="#e74c3c", lw=2.5,
                label="Test pixel")
        ax.set_title(f"{var} trajectory", fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved trajectory: {out_path}")


def plot_satellite_comparison(lat, lon, label, desc, out_path):
    """Create a 1x2 before/after satellite image."""
    print(f"  Fetching satellite imagery for Pixel {label} ...")

    img_before = get_satellite_thumbnail(lat, lon, 2016, 2017)
    img_after = get_satellite_thumbnail(lat, lon, 2022, 2023)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        f"Pixel {label}: {desc}\n"
        f"Sentinel-2 true colour  |  "
        f"lat={lat:.3f}, lon={lon:.3f}  |  "
        f"1 km \u00d7 1 km",
        fontsize=11, fontweight="bold",
    )

    axes[0].imshow(img_before)
    axes[0].set_title("2016\u20132017 median composite", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(img_after)
    axes[1].set_title("2022\u20132023 median composite", fontsize=10)
    axes[1].axis("off")

    # Add crosshair at centre
    for ax in axes:
        cx, cy = img_before.shape[1] // 2, img_before.shape[0] // 2
        ax.plot(cx, cy, "+", color="red", markersize=15, markeredgewidth=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved satellite: {out_path}")


def main():
    init_gee()

    # Cache natural/transformed per ecoregion
    eco_cache = {}

    for px in PIXELS:
        label = px["label"]
        eco_id = px["eco_id"]
        desc = px["desc"]
        print(f"\n{'='*60}")
        print(f"Pixel {label} (eco_id={eco_id}): {desc}")
        print(f"{'='*60}")

        # Load ecoregion reference (cached)
        if eco_id not in eco_cache:
            print(f"  Loading reference data for eco_id={eco_id} ...")
            nat_df = load_natural_reference(eco_id)
            trans_df = load_transformed_reference(eco_id)
            eco_cache[eco_id] = (nat_df, trans_df)
        else:
            nat_df, trans_df = eco_cache[eco_id]

        # Load test pixel
        test = load_test_pixel(
            px["pixel_id"], eco_id,
            lat=px.get("lat"), lon=px.get("lon"))
        lat = float(test["latitude"])
        lon = float(test["longitude"])
        print(f"  Loaded: pid={test['pixel_id']}, lat={lat:.4f}, "
              f"lon={lon:.4f}, sanlc={int(test['sanlc_2022'])}")

        # Find KNN
        knn_df = find_knn(lat, lon, nat_df)
        print(f"  KNN: {len(knn_df)} neighbours")

        # Trajectory plot
        traj_path = OUT_DIR / f"example_trajectory_{label.lower()}.png"
        plot_trajectory(test, nat_df, trans_df, knn_df, label, desc, traj_path)

        # Satellite before/after
        sat_path = OUT_DIR / f"example_satellite_{label.lower()}.png"
        plot_satellite_comparison(lat, lon, label, desc, sat_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
