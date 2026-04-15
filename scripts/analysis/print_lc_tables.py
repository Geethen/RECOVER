"""Print compact cluster x land cover tables for each classification option."""
import sys
import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = r"c:\Users\coach\myfiles\postdoc2\code"
RAW      = rf"{BASE_DIR}\data\abandoned_ag_gpp_2000_2022_SA.parquet"
RESULTS  = rf"{BASE_DIR}\data\trajectory_results_v2.parquet"

YEARS    = list(range(2000, 2023))
GPP_COLS = [f"GPP_{y}" for y in YEARS]
SVH_COLS = [f"SVH_{y}" for y in YEARS]

SANLC_LABELS = {
    1: "Natural",
    2: "Secondary nat.",
    3: "Artif. water",
    4: "Built-up",
    5: "Cropland",
    6: "Mine",
    7: "Plantation",
}

SAMPLE_N = 500_000


def fast_vectorized_process(y_chunk):
    B, T = y_chunk.shape
    i, j = np.triu_indices(T, k=1)
    dx   = (j - i).astype(float)
    diffs = y_chunk[:, j] - y_chunk[:, i]
    slopes    = diffs / dx[None, :]
    sen_slope = np.nanmedian(slopes, axis=1)
    signs = np.sign(diffs)
    S     = np.nansum(signs, axis=1)
    n     = np.sum(~np.isnan(y_chunk), axis=1)
    valid = n >= 5
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    sigma = np.sqrt(np.where(var_s > 0, var_s, 1))
    Z = np.zeros(B)
    Z[(S > 0) & valid] = (S[(S > 0) & valid] - 1) / sigma[(S > 0) & valid]
    Z[(S < 0) & valid] = (S[(S < 0) & valid] + 1) / sigma[(S < 0) & valid]
    p_values = 2 * stats.norm.sf(np.abs(Z))
    means = np.nanmean(y_chunk, axis=1)
    stds  = np.nanstd(y_chunk, axis=1)
    cvs   = stds / (means + 1e-6)
    return pd.DataFrame({
        "sen_slope": sen_slope, "mk_z": Z, "mk_p": p_values,
        "mean_gpp": means, "cv_gpp": cvs,
    })


def print_table(df, class_col, lc_col="lc_name", title=""):
    """Print a compact cluster x LC percentage table."""
    ct = pd.crosstab(df[class_col], df[lc_col], normalize="index") * 100
    ct["n"] = df[class_col].value_counts().reindex(ct.index)
    # reorder: n first, then LC columns sorted
    lc_cols = sorted([c for c in ct.columns if c != "n"])
    ct = ct[["n"] + lc_cols]

    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")

    # header
    hdr = f"  {'Cluster':<16} {'n':>8}"
    for lc in lc_cols:
        hdr += f"  {lc:>14}"
    print(hdr)
    print("  " + "-" * (25 + 16 * len(lc_cols)))

    for idx in sorted(ct.index, key=str):
        row = ct.loc[idx]
        line = f"  {str(idx):<16} {int(row['n']):>8}"
        for lc in lc_cols:
            line += f"  {row[lc]:>13.1f}%"
        print(line)

    print()


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=4")

    print(f"Sampling {SAMPLE_N:,} rows...")
    gpp_svh = ", ".join(GPP_COLS + SVH_COLS)
    df = con.execute(f"""
        SELECT pixel_id, latitude, longitude, eco_id, sanlc_2022,
               {gpp_svh}
        FROM '{RAW}'
        USING SAMPLE reservoir({SAMPLE_N} ROWS) REPEATABLE (42)
    """).df()

    # Eco-standardize
    gpp_arr = df[GPP_COLS].values.astype(np.float32)
    svh_arr = df[SVH_COLS].values.astype(np.float32)
    ecos    = df["eco_id"].values
    eco_stats = {}
    for eco in np.unique(ecos):
        mask = ecos == eco
        eco_stats[eco] = {
            "gpp_mean": np.nanmean(gpp_arr[mask]),
            "gpp_std":  np.nanstd(gpp_arr[mask]) + 1e-9,
            "svh_mean": np.nanmean(svh_arr[mask]),
            "svh_std":  np.nanstd(svh_arr[mask]) + 1e-9,
        }
    g_mean = np.array([eco_stats[e]["gpp_mean"] for e in ecos])[:, None]
    g_std  = np.array([eco_stats[e]["gpp_std"]  for e in ecos])[:, None]
    s_mean = np.array([eco_stats[e]["svh_mean"] for e in ecos])[:, None]
    s_std  = np.array([eco_stats[e]["svh_std"]  for e in ecos])[:, None]
    G = (gpp_arr - g_mean) / g_std
    S = (svh_arr - s_mean) / s_std

    # Trend features
    print("Computing trend features...")
    CHUNK = 100_000
    parts = []
    for k in range(0, len(G), CHUNK):
        parts.append(fast_vectorized_process(G[k:k+CHUNK]))
    feat = pd.concat(parts, ignore_index=True)
    feat["early_mean"]    = np.nanmean(G[:, :8],  axis=1)
    feat["late_mean"]     = np.nanmean(G[:, -8:], axis=1)
    feat["delta_mean"]    = feat["late_mean"] - feat["early_mean"]
    feat["autocorr_lag1"] = (
        np.nansum((G[:, 1:] - np.nanmean(G[:, 1:], axis=1, keepdims=True)) *
                  (G[:, :-1] - np.nanmean(G[:, :-1], axis=1, keepdims=True)), axis=1)
        / (np.sqrt(np.nansum((G[:, 1:]  - np.nanmean(G[:, 1:],  axis=1, keepdims=True))**2, axis=1) *
                   np.nansum((G[:, :-1] - np.nanmean(G[:, :-1], axis=1, keepdims=True))**2, axis=1)) + 1e-9)
    )
    gm = np.nanmean(G, axis=1, keepdims=True)
    sm = np.nanmean(S, axis=1, keepdims=True)
    feat["coupling_corr"] = (
        np.nansum((G - gm) * (S - sm), axis=1)
        / (np.sqrt(np.nansum((G - gm)**2, axis=1) * np.nansum((S - sm)**2, axis=1)) + 1e-9)
    )
    feat["sanlc_2022"] = df["sanlc_2022"].values
    feat["lc_name"] = feat["sanlc_2022"].map(SANLC_LABELS).fillna("Unknown")

    # --- Option A ---
    sig_pos = (feat["mk_p"] < 0.05) & (feat["sen_slope"] > 0)
    sig_neg = (feat["mk_p"] < 0.05) & (feat["sen_slope"] < 0)
    stable  = (~sig_pos) & (~sig_neg) & (feat["cv_gpp"] < 0.3)
    fluct   = (~sig_pos) & (~sig_neg) & (feat["cv_gpp"] >= 0.3)
    feat["class_A"] = "Fluctuating"
    feat.loc[stable,  "class_A"] = "Stable"
    feat.loc[sig_pos, "class_A"] = "Recovery"
    feat.loc[sig_neg, "class_A"] = "Degradation"
    print_table(feat, "class_A", title="OPTION A: Rule-Based (4 classes)")

    # --- Option C ---
    feat_cols = ["sen_slope", "mk_z", "early_mean", "late_mean", "delta_mean",
                 "cv_gpp", "autocorr_lag1", "coupling_corr"]
    X = feat[feat_cols].fillna(0).values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    for k in [5, 8, 12]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        col = f"class_C{k}"
        feat[col] = km.fit_predict(X_sc)
        print_table(feat, col, title=f"OPTION C: KMeans k={k}")

    # --- Current approach ---
    print("\nLoading current catch22 clusters...")
    curr = con.execute(f"""
        SELECT r.cluster, s.sanlc_2022
        FROM '{RESULTS}' r
        JOIN '{RAW}' s
          ON CAST(s.latitude  AS FLOAT) = CAST(r.latitude  AS FLOAT)
         AND CAST(s.longitude AS FLOAT) = CAST(r.longitude AS FLOAT)
        USING SAMPLE reservoir(500000 ROWS) REPEATABLE(42)
    """).df()
    curr["lc_name"] = curr["sanlc_2022"].map(SANLC_LABELS).fillna("Unknown")
    print_table(curr, "cluster", title="CURRENT: catch22 + UMAP + HDBSCAN")

    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
