"""
Plot example GPP and SVH time series for each trajectory category
(Recovery, Stable, Degradation) across ecoregions.

For each ecoregion, show 3 panels (Recovery, Stable, Degradation) with
5 random example trajectories each, for both GPP and SVH.
"""
import sys
import numpy as np
import pandas as pd
import duckdb
import pyarrow.parquet as pq
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
RAW = BASE_DIR / "data" / "abandoned_ag_gpp_2000_2022_SA.parquet"
ECO_CSV = BASE_DIR / "data" / "ecoregion_sds.csv"
OUT_DIR = BASE_DIR / "plots"
YEARS = list(range(2000, 2023))
GPP_COLS = [f"GPP_{y}" for y in YEARS]
SVH_COLS = [f"SVH_{y}" for y in YEARS]
NATURAL_CODES = [1, 2]
N_EXAMPLES = 5
SAMPLE_PER_ECO = 20_000  # sample per ecoregion for classification

# Top ecoregions by pixel count (skip tiny ones)
TOP_ECOS = [81, 41, 38, 40, 97, 89, 48, 90, 101, 88, 16, 19, 15, 116]

CLASS_COLORS = {"Recovery": "#2ecc71", "Stable": "#3498db", "Degradation": "#e74c3c"}


def load_eco_names():
    df = pd.read_csv(ECO_CSV, usecols=["ECO_ID", "ECO_NAME"]).drop_duplicates("ECO_ID")
    return dict(zip(df["ECO_ID"].astype(int), df["ECO_NAME"]))


def mk_sen_batch(y_arr):
    B, T = y_arr.shape
    i, j = np.triu_indices(T, k=1)
    dx = (j - i).astype(np.float32)
    diffs = y_arr[:, j] - y_arr[:, i]
    slopes = diffs / dx[None, :]
    sen_slope = np.median(slopes, axis=1).astype(np.float32)
    S = np.sum(np.sign(diffs), axis=1).astype(np.float64)
    del diffs, slopes
    var_s = (T * (T - 1) * (2 * T + 5)) / 18.0
    sigma = np.sqrt(var_s)
    Z = np.zeros(B)
    pos = S > 0
    neg = S < 0
    Z[pos] = (S[pos] - 1) / sigma
    Z[neg] = (S[neg] + 1) / sigma
    p = 2 * stats.norm.sf(np.abs(Z))
    return sen_slope, Z, p


def classify_trend(slope, p, alpha=0.05):
    sig_pos = (p < alpha) & (slope > 0)
    sig_neg = (p < alpha) & (slope < 0)
    labels = np.full(len(slope), "Stable", dtype=object)
    labels[sig_pos] = "Recovery"
    labels[sig_neg] = "Degradation"
    return labels


def main():
    eco_names = load_eco_names()
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=4")

    eco_str = ", ".join(str(e) for e in TOP_ECOS)
    nat_str = ", ".join(str(c) for c in NATURAL_CODES)
    gpp_svh = ", ".join(GPP_COLS + SVH_COLS)

    # Sample data per ecoregion
    print("Sampling data per ecoregion...")
    df = con.execute(f"""
        SELECT eco_id, {gpp_svh}
        FROM '{RAW}'
        WHERE sanlc_2022 IN ({nat_str})
          AND eco_id IN ({eco_str})
        USING SAMPLE reservoir(200000 ROWS) REPEATABLE(42)
    """).df()
    con.close()
    print(f"  Loaded {len(df):,} rows across {df['eco_id'].nunique()} ecoregions")

    # Classify each pixel
    print("Computing trends...")
    gpp_raw = df[GPP_COLS].values.astype(np.float32)
    svh_raw = df[SVH_COLS].values.astype(np.float32)
    ecos = df["eco_id"].values

    # Eco-standardize for classification
    gpp_z = gpp_raw.copy()
    svh_z = svh_raw.copy()
    for eco in np.unique(ecos):
        m = ecos == eco
        gpp_z[m] = (gpp_z[m] - np.nanmean(gpp_z[m])) / (np.nanstd(gpp_z[m]) + 1e-9)
        svh_z[m] = (svh_z[m] - np.nanmean(svh_z[m])) / (np.nanstd(svh_z[m]) + 1e-9)

    # Compute trends in batches
    B = len(gpp_z)
    SB = 1000
    gpp_slope = np.empty(B, dtype=np.float32)
    gpp_p = np.empty(B, dtype=np.float64)
    svh_slope = np.empty(B, dtype=np.float32)
    svh_p = np.empty(B, dtype=np.float64)

    for s in range(0, B, SB):
        e = min(s + SB, B)
        gs, _, gp = mk_sen_batch(gpp_z[s:e])
        ss, _, sp = mk_sen_batch(svh_z[s:e])
        gpp_slope[s:e] = gs
        gpp_p[s:e] = gp
        svh_slope[s:e] = ss
        svh_p[s:e] = sp

    df["gpp_class"] = classify_trend(gpp_slope, gpp_p)
    df["svh_class"] = classify_trend(svh_slope, svh_p)
    df["gpp_slope"] = gpp_slope
    df["svh_slope"] = svh_slope

    # Store raw values back for plotting
    for i, col in enumerate(GPP_COLS):
        df[f"gpp_raw_{i}"] = gpp_raw[:, i]
    for i, col in enumerate(SVH_COLS):
        df[f"svh_raw_{i}"] = svh_raw[:, i]

    gpp_raw_cols = [f"gpp_raw_{i}" for i in range(len(YEARS))]
    svh_raw_cols = [f"svh_raw_{i}" for i in range(len(YEARS))]

    # ── Figure 1: GPP trajectories by ecoregion and class ──
    print("Plotting GPP trajectories...")
    classes = ["Recovery", "Stable", "Degradation"]
    n_ecos = len(TOP_ECOS)
    n_cols = 3  # Recovery, Stable, Degradation

    fig, axes = plt.subplots(n_ecos, n_cols, figsize=(16, 3 * n_ecos), sharey=False)

    for row, eco_id in enumerate(TOP_ECOS):
        eco_data = df[df["eco_id"] == eco_id]
        eco_name = eco_names.get(eco_id, f"Eco {eco_id}")

        for col, cls in enumerate(classes):
            ax = axes[row, col]
            subset = eco_data[eco_data["gpp_class"] == cls]

            if len(subset) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
            else:
                # Pick N_EXAMPLES sorted by slope magnitude (show representative ones)
                if cls == "Recovery":
                    examples = subset.nlargest(N_EXAMPLES, "gpp_slope")
                elif cls == "Degradation":
                    examples = subset.nsmallest(N_EXAMPLES, "gpp_slope")
                else:
                    examples = subset.iloc[np.random.RandomState(42).choice(len(subset), min(N_EXAMPLES, len(subset)), replace=False)]

                for _, px in examples.iterrows():
                    vals = px[gpp_raw_cols].values.astype(float)
                    ax.plot(YEARS, vals, color=CLASS_COLORS[cls], alpha=0.5, linewidth=0.8)

                # Plot mean of all pixels in this class
                mean_vals = subset[gpp_raw_cols].mean().values
                ax.plot(YEARS, mean_vals, color=CLASS_COLORS[cls], linewidth=2.5, linestyle="--", label=f"Mean (n={len(subset):,})")

            if row == 0:
                ax.set_title(cls, fontsize=13, fontweight="bold", color=CLASS_COLORS[cls])
            if col == 0:
                ax.set_ylabel(f"{eco_name}\n(n={len(eco_data):,})", fontsize=8)
            ax.tick_params(labelsize=7)
            if row < n_ecos - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Year", fontsize=8)
                ax.tick_params(axis="x", rotation=45)
            ax.legend(fontsize=6, loc="upper left") if len(subset) > 0 else None

    fig.suptitle("GPP Trajectories by Ecoregion and Trend Class\n(raw values, 5 examples + class mean)", fontsize=14, y=1.01)
    plt.tight_layout()
    out1 = OUT_DIR / "trajectory_examples_gpp.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out1}")
    plt.close(fig)

    # ── Figure 2: SVH trajectories by ecoregion and class ──
    print("Plotting SVH trajectories...")
    fig2, axes2 = plt.subplots(n_ecos, n_cols, figsize=(16, 3 * n_ecos), sharey=False)

    for row, eco_id in enumerate(TOP_ECOS):
        eco_data = df[df["eco_id"] == eco_id]
        eco_name = eco_names.get(eco_id, f"Eco {eco_id}")

        for col, cls in enumerate(classes):
            ax = axes2[row, col]
            subset = eco_data[eco_data["svh_class"] == cls]

            if len(subset) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
            else:
                if cls == "Recovery":
                    examples = subset.nlargest(N_EXAMPLES, "svh_slope")
                elif cls == "Degradation":
                    examples = subset.nsmallest(N_EXAMPLES, "svh_slope")
                else:
                    examples = subset.iloc[np.random.RandomState(42).choice(len(subset), min(N_EXAMPLES, len(subset)), replace=False)]

                for _, px in examples.iterrows():
                    vals = px[svh_raw_cols].values.astype(float)
                    ax.plot(YEARS, vals, color=CLASS_COLORS[cls], alpha=0.5, linewidth=0.8)

                mean_vals = subset[svh_raw_cols].mean().values
                ax.plot(YEARS, mean_vals, color=CLASS_COLORS[cls], linewidth=2.5, linestyle="--", label=f"Mean (n={len(subset):,})")

            if row == 0:
                ax.set_title(cls, fontsize=13, fontweight="bold", color=CLASS_COLORS[cls])
            if col == 0:
                ax.set_ylabel(f"{eco_name}\n(n={len(eco_data):,})", fontsize=8)
            ax.tick_params(labelsize=7)
            if row < n_ecos - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Year", fontsize=8)
                ax.tick_params(axis="x", rotation=45)
            ax.legend(fontsize=6, loc="upper left") if len(subset) > 0 else None

    fig2.suptitle("SVH Trajectories by Ecoregion and Trend Class\n(raw values, 5 examples + class mean)", fontsize=14, y=1.01)
    plt.tight_layout()
    out2 = OUT_DIR / "trajectory_examples_svh.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out2}")
    plt.close(fig2)

    # ── Figure 3: Combined GPP+SVH side-by-side for select ecoregions ──
    print("Plotting combined GPP+SVH for select ecoregions...")
    select_ecos = [16, 81, 41, 89, 38, 48]  # diverse set
    fig3, axes3 = plt.subplots(len(select_ecos), 6, figsize=(24, 3.5 * len(select_ecos)))

    for row, eco_id in enumerate(select_ecos):
        eco_data = df[df["eco_id"] == eco_id]
        eco_name = eco_names.get(eco_id, f"Eco {eco_id}")

        for ci, cls in enumerate(classes):
            # GPP column
            ax_gpp = axes3[row, ci]
            subset_gpp = eco_data[eco_data["gpp_class"] == cls]
            if len(subset_gpp) > 0:
                if cls == "Recovery":
                    ex = subset_gpp.nlargest(N_EXAMPLES, "gpp_slope")
                elif cls == "Degradation":
                    ex = subset_gpp.nsmallest(N_EXAMPLES, "gpp_slope")
                else:
                    ex = subset_gpp.iloc[np.random.RandomState(42).choice(len(subset_gpp), min(N_EXAMPLES, len(subset_gpp)), replace=False)]
                for _, px in ex.iterrows():
                    ax_gpp.plot(YEARS, px[gpp_raw_cols].values.astype(float), color=CLASS_COLORS[cls], alpha=0.4, linewidth=0.7)
                ax_gpp.plot(YEARS, subset_gpp[gpp_raw_cols].mean().values, color=CLASS_COLORS[cls], linewidth=2.5, linestyle="--")
            else:
                ax_gpp.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_gpp.transAxes, color="gray")

            # SVH column
            ax_svh = axes3[row, ci + 3]
            subset_svh = eco_data[eco_data["svh_class"] == cls]
            if len(subset_svh) > 0:
                if cls == "Recovery":
                    ex = subset_svh.nlargest(N_EXAMPLES, "svh_slope")
                elif cls == "Degradation":
                    ex = subset_svh.nsmallest(N_EXAMPLES, "svh_slope")
                else:
                    ex = subset_svh.iloc[np.random.RandomState(42).choice(len(subset_svh), min(N_EXAMPLES, len(subset_svh)), replace=False)]
                for _, px in ex.iterrows():
                    ax_svh.plot(YEARS, px[svh_raw_cols].values.astype(float), color=CLASS_COLORS[cls], alpha=0.4, linewidth=0.7)
                ax_svh.plot(YEARS, subset_svh[svh_raw_cols].mean().values, color=CLASS_COLORS[cls], linewidth=2.5, linestyle="--")
            else:
                ax_svh.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_svh.transAxes, color="gray")

            if row == 0:
                ax_gpp.set_title(f"GPP - {cls}", fontsize=10, fontweight="bold", color=CLASS_COLORS[cls])
                ax_svh.set_title(f"SVH - {cls}", fontsize=10, fontweight="bold", color=CLASS_COLORS[cls])
            if ci == 0:
                ax_gpp.set_ylabel(eco_name, fontsize=9, fontweight="bold")

            for ax in [ax_gpp, ax_svh]:
                ax.tick_params(labelsize=6)
                if row < len(select_ecos) - 1:
                    ax.set_xticklabels([])
                else:
                    ax.tick_params(axis="x", rotation=45)

    fig3.suptitle("GPP and SVH Trajectories by Ecoregion and Trend Class\n(5 examples + class mean dashed)", fontsize=14, y=1.01)
    plt.tight_layout()
    out3 = OUT_DIR / "trajectory_examples_combined.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out3}")
    plt.close(fig3)

    print("Done.")


if __name__ == "__main__":
    main()