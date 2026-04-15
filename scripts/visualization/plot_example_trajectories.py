"""
Generate trajectory plots for three example pixels (A, B, C) used in the
recovery_assessment_guide.md, in the same style as the comparison plot.

Each pixel gets a 1x2 figure: GPP trajectory | SVH trajectory
with natural mean +/- 1SD envelope, transformed mean, KNN mean, test pixel.
"""
import sys
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from sklearn.neighbors import BallTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
ABANDONED_AG = BASE_DIR / "data" / "abandoned_ag_gpp_2000_2022_SA.parquet"
INDICES_GPP = BASE_DIR / "data" / "indices_gpp_svh_2000_2022.parquet"
NAT_SUBSET = BASE_DIR / "data" / "dfsubsetNatural.parquet"
OUT_DIR = BASE_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

TARGET_ECO = 81
YEARS = list(range(2000, 2023))
GPP_COLS = [f"GPP_{y}" for y in YEARS]
SVH_COLS = [f"SVH_{y}" for y in YEARS]
SVH_SCALE = 0.1  # raw SVH values must be multiplied by 0.1
KNN_K = 10
KNN_RADIUS_KM = 50.0
EARTH_R_KM = 6371.0

PIXELS = [
    {"pid": 8393,  "label": "A", "desc": "Moderate recovery (score 53.6)"},
    {"pid": 34240, "label": "B", "desc": "Advanced recovery (score 82.2)"},
    {"pid": 79489, "label": "C", "desc": "Early/limited recovery (score 24.1)"},
]


def load_natural_reference():
    """Load natural reference with GPP/SVH time series for eco_id=81."""
    gpp_sel = ", ".join(f"g.{c}" for c in GPP_COLS)
    svh_sel = ", ".join(f"g.{c}" for c in SVH_COLS)
    sql = f"""
        SELECT g.latitude, g.longitude, {gpp_sel}, {svh_sel}
        FROM '{NAT_SUBSET}' n
        JOIN '{INDICES_GPP}' g
          ON n.id = split_part(g.pixel_id, '_', 1)
        WHERE g.eco_id = {TARGET_ECO}
    """
    df = duckdb.sql(sql).df()
    print(f"Natural reference: {len(df):,} pixels (eco_id={TARGET_ECO})")
    return df


def load_transformed_reference(n=5000):
    """Load transformed reference with GPP/SVH time series for eco_id=81."""
    sql = f"""
        SELECT latitude, longitude, {", ".join(GPP_COLS)}, {", ".join(SVH_COLS)}
        FROM '{INDICES_GPP}'
        WHERE sanlc_2022 NOT IN (1, 2) AND eco_id = {TARGET_ECO}
        USING SAMPLE reservoir({n} ROWS) REPEATABLE(42)
    """
    df = duckdb.sql(sql).df()
    print(f"Transformed reference: {len(df):,} pixels")
    return df


def load_test_pixel(pixel_id):
    """Load a single test pixel time series from abandoned_ag."""
    gpp_sel = ", ".join(GPP_COLS)
    svh_sel = ", ".join(SVH_COLS)
    sql = f"""
        SELECT pixel_id, latitude, longitude, eco_id, sanlc_2022,
               {gpp_sel}, {svh_sel}
        FROM '{ABANDONED_AG}'
        WHERE pixel_id = {pixel_id}
        LIMIT 1
    """
    df = duckdb.sql(sql).df()
    if len(df) == 0:
        raise RuntimeError(f"Pixel {pixel_id} not found")
    return df.iloc[0]


def find_knn(test_lat, test_lon, nat_df, k=KNN_K, radius_km=KNN_RADIUS_KM):
    """Find K nearest natural neighbours within radius."""
    coords_rad = np.radians(nat_df[["latitude", "longitude"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    test_rad = np.radians([[test_lat, test_lon]])
    radius_rad = radius_km / EARTH_R_KM

    idx = tree.query_radius(test_rad, r=radius_rad)[0]
    if len(idx) < k:
        _, idx = tree.query(test_rad, k=min(k, len(nat_df)))
        idx = idx[0]
    else:
        dists, all_idx = tree.query(test_rad, k=len(nat_df))
        within = [i for i in all_idx[0] if i in set(idx)]
        d_within = [dists[0][list(all_idx[0]).index(i)] for i in within]
        order = np.argsort(d_within)[:k]
        idx = [within[o] for o in order]

    return nat_df.iloc[idx]


def plot_pixel(test, nat_df, trans_df, knn_df, label, desc, out_path):
    """Create a 1x2 trajectory plot for a single pixel."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Pixel {label}: {desc}\n"
        f"pixel_id={test['pixel_id']}  |  "
        f"lat={test['latitude']:.3f}, lon={test['longitude']:.3f}  |  "
        f"SANLC={int(test['sanlc_2022'])}",
        fontsize=11, fontweight="bold",
    )

    nat_gpp = nat_df[GPP_COLS].values.astype(float)
    nat_svh = nat_df[SVH_COLS].values.astype(float) * SVH_SCALE
    trans_gpp = trans_df[GPP_COLS].values.astype(float)
    trans_svh = trans_df[SVH_COLS].values.astype(float) * SVH_SCALE
    knn_gpp_mean = knn_df[GPP_COLS].values.astype(float).mean(axis=0)
    knn_svh_mean = knn_df[SVH_COLS].values.astype(float).mean(axis=0) * SVH_SCALE
    test_gpp = test[GPP_COLS].values.astype(float)
    test_svh = test[SVH_COLS].values.astype(float) * SVH_SCALE

    # GPP trajectory
    ax = axes[0]
    nat_mean = nat_gpp.mean(axis=0)
    nat_std = nat_gpp.std(axis=0)
    ax.fill_between(YEARS, nat_mean - nat_std, nat_mean + nat_std,
                     alpha=0.2, color="#2ecc71", label="Natural +/-1 SD")
    ax.plot(YEARS, nat_mean, "--", color="#2ecc71", lw=1.5, label="Natural mean")
    ax.plot(YEARS, trans_gpp.mean(axis=0), "--", color="#e67e22", lw=1.5,
            label="Transformed mean")
    ax.plot(YEARS, knn_gpp_mean, "--", color="#3498db", lw=1.5,
            label=f"KNN mean (K={len(knn_df)})")
    ax.plot(YEARS, test_gpp, "-", color="#e74c3c", lw=2.5, label="Test pixel")
    ax.set_title("GPP trajectory", fontsize=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("GPP (gC/m\u00b2/yr)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # SVH trajectory
    ax = axes[1]
    nat_mean = nat_svh.mean(axis=0)
    nat_std = nat_svh.std(axis=0)
    ax.fill_between(YEARS, nat_mean - nat_std, nat_mean + nat_std,
                     alpha=0.2, color="#2ecc71", label="Natural +/-1 SD")
    ax.plot(YEARS, nat_mean, "--", color="#2ecc71", lw=1.5, label="Natural mean")
    ax.plot(YEARS, trans_svh.mean(axis=0), "--", color="#e67e22", lw=1.5,
            label="Transformed mean")
    ax.plot(YEARS, knn_svh_mean, "--", color="#3498db", lw=1.5,
            label=f"KNN mean (K={len(knn_df)})")
    ax.plot(YEARS, test_svh, "-", color="#e74c3c", lw=2.5, label="Test pixel")
    ax.set_title("SVH trajectory", fontsize=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("SVH (m)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    print("Loading reference data ...")
    nat_df = load_natural_reference()
    trans_df = load_transformed_reference()

    for px in PIXELS:
        pid = px["pid"]
        label = px["label"]
        desc = px["desc"]
        print(f"\nPixel {label} (pid={pid}): {desc}")

        test = load_test_pixel(pid)
        knn_df = find_knn(test["latitude"], test["longitude"], nat_df)
        print(f"  KNN: {len(knn_df)} neighbours found")

        out_path = OUT_DIR / f"example_trajectory_{label.lower()}.png"
        plot_pixel(test, nat_df, trans_df, knn_df, label, desc, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
