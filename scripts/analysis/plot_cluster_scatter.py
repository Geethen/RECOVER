"""Scatter plots of PCA components colored by land cover category."""
import numpy as np
import pandas as pd
import duckdb
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

BASE_DIR = r"c:\Users\coach\myfiles\postdoc2\code"
RAW = rf"{BASE_DIR}\data\abandoned_ag_gpp_2000_2022_SA.parquet"
YEARS = list(range(2000, 2023))
GPP_COLS = [f"GPP_{y}" for y in YEARS]
SVH_COLS = [f"SVH_{y}" for y in YEARS]
SAMPLE_N = 200_000
OUT_DIR = rf"{BASE_DIR}\plots"

SANLC_LABELS = {
    1: "Natural", 2: "Secondary nat.", 3: "Artif. water",
    4: "Built-up", 5: "Cropland", 6: "Mine", 7: "Plantation",
}
LC_COLORS = {
    "Natural": "#228B22", "Secondary nat.": "#90EE90", "Artif. water": "#4169E1",
    "Built-up": "#DC143C", "Cropland": "#FFD700", "Mine": "#8B008B", "Plantation": "#FF8C00",
}


def mk_process(y_chunk):
    B, T = y_chunk.shape
    i, j = np.triu_indices(T, k=1)
    dx = (j - i).astype(float)
    diffs = y_chunk[:, j] - y_chunk[:, i]
    slopes = diffs / dx[None, :]
    sen_slope = np.nanmedian(slopes, axis=1)
    signs = np.sign(diffs)
    S = np.nansum(signs, axis=1)
    n = np.sum(~np.isnan(y_chunk), axis=1)
    valid = n >= 5
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    sigma = np.sqrt(np.where(var_s > 0, var_s, 1))
    Z = np.zeros(B)
    Z[(S > 0) & valid] = (S[(S > 0) & valid] - 1) / sigma[(S > 0) & valid]
    Z[(S < 0) & valid] = (S[(S < 0) & valid] + 1) / sigma[(S < 0) & valid]
    p = 2 * stats.norm.sf(np.abs(Z))
    return pd.DataFrame({
        "sen_slope": sen_slope, "mk_z": Z, "mk_p": p,
        "mean_gpp": np.nanmean(y_chunk, axis=1),
        "cv_gpp": np.nanstd(y_chunk, axis=1) / (np.nanmean(y_chunk, axis=1) + 1e-6),
    })


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=4")

    print(f"Sampling {SAMPLE_N:,} rows...")
    gpp_svh = ", ".join(GPP_COLS + SVH_COLS)
    df = con.execute(f"""
        SELECT pixel_id, latitude, longitude, eco_id, sanlc_2022, {gpp_svh}
        FROM '{RAW}'
        USING SAMPLE reservoir({SAMPLE_N} ROWS) REPEATABLE (42)
    """).df()
    con.close()

    # Eco-standardize
    gpp_arr = df[GPP_COLS].values.astype(np.float32)
    svh_arr = df[SVH_COLS].values.astype(np.float32)
    ecos = df["eco_id"].values
    eco_stats = {}
    for eco in np.unique(ecos):
        m = ecos == eco
        eco_stats[eco] = {
            "gm": np.nanmean(gpp_arr[m]), "gs": np.nanstd(gpp_arr[m]) + 1e-9,
            "sm": np.nanmean(svh_arr[m]), "ss": np.nanstd(svh_arr[m]) + 1e-9,
        }
    gm = np.array([eco_stats[e]["gm"] for e in ecos])[:, None]
    gs = np.array([eco_stats[e]["gs"] for e in ecos])[:, None]
    sm = np.array([eco_stats[e]["sm"] for e in ecos])[:, None]
    ss = np.array([eco_stats[e]["ss"] for e in ecos])[:, None]
    G = (gpp_arr - gm) / gs
    S = (svh_arr - sm) / ss

    # Trend features
    print("Computing trend features...")
    parts = []
    for k in range(0, len(G), 50_000):
        parts.append(mk_process(G[k:k + 50_000]))
    feat = pd.concat(parts, ignore_index=True)
    feat["early_mean"] = np.nanmean(G[:, :8], axis=1)
    feat["late_mean"] = np.nanmean(G[:, -8:], axis=1)
    feat["delta_mean"] = feat["late_mean"] - feat["early_mean"]
    feat["autocorr_lag1"] = (
        np.nansum((G[:, 1:] - np.nanmean(G[:, 1:], axis=1, keepdims=True)) *
                  (G[:, :-1] - np.nanmean(G[:, :-1], axis=1, keepdims=True)), axis=1)
        / (np.sqrt(np.nansum((G[:, 1:] - np.nanmean(G[:, 1:], axis=1, keepdims=True)) ** 2, axis=1) *
                   np.nansum((G[:, :-1] - np.nanmean(G[:, :-1], axis=1, keepdims=True)) ** 2, axis=1)) + 1e-9)
    )
    gmm = np.nanmean(G, axis=1, keepdims=True)
    smm = np.nanmean(S, axis=1, keepdims=True)
    feat["coupling_corr"] = (
        np.nansum((G - gmm) * (S - smm), axis=1)
        / (np.sqrt(np.nansum((G - gmm) ** 2, axis=1) * np.nansum((S - smm) ** 2, axis=1)) + 1e-9)
    )

    feat["sanlc_2022"] = df["sanlc_2022"].values
    feat["lc_name"] = feat["sanlc_2022"].map(SANLC_LABELS).fillna("Unknown")

    # Classifications
    sig_pos = (feat["mk_p"] < 0.05) & (feat["sen_slope"] > 0)
    sig_neg = (feat["mk_p"] < 0.05) & (feat["sen_slope"] < 0)
    stable = (~sig_pos) & (~sig_neg) & (feat["cv_gpp"] < 0.3)
    feat["Option_A"] = "Fluctuating"
    feat.loc[stable, "Option_A"] = "Stable"
    feat.loc[sig_pos, "Option_A"] = "Recovery"
    feat.loc[sig_neg, "Option_A"] = "Degradation"

    feat_cols = ["sen_slope", "mk_z", "early_mean", "late_mean", "delta_mean",
                 "cv_gpp", "autocorr_lag1", "coupling_corr"]
    X = feat[feat_cols].fillna(0).values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    for k in [5, 8, 12]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        feat[f"C_k{k}"] = km.fit_predict(X_sc)

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X_sc)
    feat["PC1"] = pcs[:, 0]
    feat["PC2"] = pcs[:, 1]
    feat["PC3"] = pcs[:, 2]
    ev = pca.explained_variance_ratio_
    print(f"  Explained variance: PC1={ev[0]:.1%}, PC2={ev[1]:.1%}, PC3={ev[2]:.1%}")

    # Subsample for plotting
    np.random.seed(42)
    plot_idx = np.random.choice(len(feat), size=50_000, replace=False)
    fp = feat.iloc[plot_idx]

    lc_order = ["Secondary nat.", "Cropland", "Natural", "Plantation",
                "Built-up", "Mine", "Artif. water"]

    def scatter_by_lc(ax, fp, xvar, yvar):
        """Plot scatter colored by land cover, rare classes on top."""
        for lc in lc_order:
            mask = fp["lc_name"] == lc
            if mask.sum() == 0:
                continue
            ax.scatter(fp.loc[mask, xvar], fp.loc[mask, yvar],
                       c=[LC_COLORS[lc]], s=2, alpha=0.4, label=lc, rasterized=True)

    # --- Figure 1: PCA scatter colored by LC ---
    print("Plotting...")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    scatter_by_lc(axes[0], fp, "PC1", "PC2")
    axes[0].set_xlabel(f"PC1 ({ev[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({ev[1]:.1%})")
    axes[0].set_title("PC1 vs PC2")
    axes[0].legend(markerscale=4, fontsize=8, loc="upper right")

    scatter_by_lc(axes[1], fp, "PC1", "PC3")
    axes[1].set_xlabel(f"PC1 ({ev[0]:.1%})")
    axes[1].set_ylabel(f"PC3 ({ev[2]:.1%})")
    axes[1].set_title("PC1 vs PC3")

    scatter_by_lc(axes[2], fp, "PC2", "PC3")
    axes[2].set_xlabel(f"PC2 ({ev[1]:.1%})")
    axes[2].set_ylabel(f"PC3 ({ev[2]:.1%})")
    axes[2].set_title("PC2 vs PC3")

    plt.suptitle("Trend Feature PCA - Colored by Land Cover (50k subsample)", fontsize=14, y=1.02)
    plt.tight_layout()
    out1 = rf"{OUT_DIR}\pca_scatter_by_landcover.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close()

    # --- Figure 2: Raw feature pairs colored by LC ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
    pairs = [
        ("sen_slope", "delta_mean"),
        ("sen_slope", "cv_gpp"),
        ("sen_slope", "coupling_corr"),
        ("delta_mean", "cv_gpp"),
        ("autocorr_lag1", "coupling_corr"),
        ("early_mean", "late_mean"),
    ]

    for ax, (xvar, yvar) in zip(axes2.flat, pairs):
        scatter_by_lc(ax, fp, xvar, yvar)
        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)
        if xvar == "cv_gpp":
            ax.set_xlim(-5, 5)
        if yvar == "cv_gpp":
            ax.set_ylim(-5, 5)
        ax.legend(markerscale=4, fontsize=7, loc="upper right")

    plt.suptitle("Raw Feature Pairs - Colored by Land Cover (50k subsample)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = rf"{OUT_DIR}\feature_scatter_by_landcover.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()

    # --- Figure 3: Per-class PCA facets colored by LC ---
    # Show how LC distributes within each Option A class
    classes_A = ["Recovery", "Stable", "Fluctuating", "Degradation"]
    fig3, axes3 = plt.subplots(1, 4, figsize=(24, 6))

    for ax, cls in zip(axes3, classes_A):
        sub = fp[fp["Option_A"] == cls]
        scatter_by_lc(ax, sub, "PC1", "PC2")
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.set_title(f"{cls} (n={len(sub):,})")
        ax.legend(markerscale=4, fontsize=7, loc="upper right")

    plt.suptitle("Option A Classes - PCA Colored by Land Cover (50k subsample)", fontsize=14, y=1.02)
    plt.tight_layout()
    out3 = rf"{OUT_DIR}\option_a_pca_by_landcover.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved: {out3}")
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
