"""
Test trend-based trajectory classification approaches on a 500k sample.
Compares:
  Option A - Rule-based (Mann-Kendall + Theil-Sen thresholds)
  Option C - KMeans on directional trend features

vs. current catch22 UMAP+HDBSCAN approach (reference).
"""
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

YEARS     = list(range(2000, 2023))
GPP_COLS  = [f"GPP_{y}" for y in YEARS]
SVH_COLS  = [f"SVH_{y}" for y in YEARS]
EARLY     = [f"GPP_{y}" for y in range(2000, 2008)]   # first 8 yrs
LATE      = [f"GPP_{y}" for y in range(2015, 2023)]   # last 8 yrs

SANLC_LABELS = {
    1: "Natural/near-natural",
    2: "Secondary natural",
    3: "Artificial water",
    4: "Built-up",
    5: "Cropland",
    6: "Mine",
    7: "Plantation",
}

SAMPLE_N = 500_000


# ── Helpers ──────────────────────────────────────────────────────────────────

def fast_vectorized_process(y_chunk):
    """Vectorized Mann-Kendall + Theil-Sen (adapted from analyze_trajectories.py)."""
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
        "sen_slope": sen_slope,
        "mk_z":      Z,
        "mk_p":      p_values,
        "mean_gpp":  means,
        "cv_gpp":    cvs,
    })


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def lc_crosstab(df, class_col, label=""):
    """Print full land cover tables: (1) LC within each class, (2) class within each LC."""
    df = df.copy()
    df["lc_name"] = df["sanlc_2022"].map(SANLC_LABELS).fillna("Unknown")
    all_lc  = sorted(df["lc_name"].unique())
    all_cls = sorted(df[class_col].unique(), key=str)

    # ── View 1: land cover % within each class ──
    print(f"\n--- {label}: land cover % within each class ---")
    hdr = f"  {'Class':<22} {'n':>8} |"
    for lc in all_lc:
        hdr += f" {lc[:14]:>14}"
    print(hdr)
    print("  " + "-" * (31 + 15 * len(all_lc)))

    purities = []
    for cls in all_cls:
        grp = df[df[class_col] == cls]
        total = len(grp)
        lc_pct = grp["lc_name"].value_counts(normalize=True) * 100
        purities.append(lc_pct.iloc[0])
        row = f"  {str(cls):<22} {total:>8,} |"
        for lc in all_lc:
            pct = lc_pct.get(lc, 0.0)
            row += f" {pct:>13.1f}%"
        print(row)

    print(f"\n  Avg dominant-LC purity: {np.mean(purities):.1f}%  "
          f"(lower = better separation)")

    # ── View 2: class % within each land cover ──
    print(f"\n--- {label}: class distribution within each land cover ---")
    hdr2 = f"  {'Land cover':<22} {'n':>8} |"
    for cls in all_cls:
        hdr2 += f" {str(cls):>14}"
    print(hdr2)
    print("  " + "-" * (31 + 15 * len(all_cls)))

    for lc in all_lc:
        grp = df[df["lc_name"] == lc]
        total = len(grp)
        cls_pct = grp[class_col].value_counts(normalize=True) * 100
        row = f"  {lc[:22]:<22} {total:>8,} |"
        for cls in all_cls:
            pct = cls_pct.get(cls, 0.0)
            row += f" {pct:>13.1f}%"
        print(row)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=4")

    # ── 1. Sample ────────────────────────────────────────────────────────────
    print(f"Sampling {SAMPLE_N:,} rows from raw data...")
    gpp_svh = ", ".join(GPP_COLS + SVH_COLS)
    df = con.execute(f"""
        SELECT pixel_id, latitude, longitude, eco_id, sanlc_2022,
               {gpp_svh}
        FROM '{RAW}'
        USING SAMPLE reservoir({SAMPLE_N} ROWS) REPEATABLE (42)
    """).df()
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # ── 2. Eco-standardize ───────────────────────────────────────────────────
    print("Computing per-ecoregion standardization...")
    gpp_arr = df[GPP_COLS].values.astype(np.float32)
    svh_arr = df[SVH_COLS].values.astype(np.float32)
    ecos    = df["eco_id"].values

    eco_stats = {}
    for eco in np.unique(ecos):
        mask = ecos == eco
        eco_stats[eco] = {
            "gpp_mean": np.nanmean(gpp_arr[mask]),
            "gpp_std":  np.nanstd(gpp_arr[mask])  + 1e-9,
            "svh_mean": np.nanmean(svh_arr[mask]),
            "svh_std":  np.nanstd(svh_arr[mask])  + 1e-9,
        }

    g_mean = np.array([eco_stats[e]["gpp_mean"] for e in ecos])[:, None]
    g_std  = np.array([eco_stats[e]["gpp_std"]  for e in ecos])[:, None]
    s_mean = np.array([eco_stats[e]["svh_mean"] for e in ecos])[:, None]
    s_std  = np.array([eco_stats[e]["svh_std"]  for e in ecos])[:, None]

    G = (gpp_arr - g_mean) / g_std
    S = (svh_arr - s_mean) / s_std

    # ── 3. Trend features ────────────────────────────────────────────────────
    print("Computing trend features (vectorized)...")
    CHUNK = 100_000
    parts = []
    for k in range(0, len(G), CHUNK):
        parts.append(fast_vectorized_process(G[k:k+CHUNK]))
    feat = pd.concat(parts, ignore_index=True)

    # Additional directional features
    feat["early_mean"]    = np.nanmean(G[:, :8],  axis=1)   # 2000-2007
    feat["late_mean"]     = np.nanmean(G[:, -8:], axis=1)   # 2015-2022
    feat["delta_mean"]    = feat["late_mean"] - feat["early_mean"]
    feat["autocorr_lag1"] = (
        np.nansum((G[:, 1:] - np.nanmean(G[:, 1:], axis=1, keepdims=True)) *
                  (G[:, :-1] - np.nanmean(G[:, :-1], axis=1, keepdims=True)), axis=1)
        / (np.sqrt(np.nansum((G[:, 1:]  - np.nanmean(G[:, 1:],  axis=1, keepdims=True))**2, axis=1) *
                   np.nansum((G[:, :-1] - np.nanmean(G[:, :-1], axis=1, keepdims=True))**2, axis=1)) + 1e-9)
    )
    # GPP-SVH coupling
    gm = np.nanmean(G, axis=1, keepdims=True)
    sm = np.nanmean(S, axis=1, keepdims=True)
    feat["coupling_corr"] = (
        np.nansum((G - gm) * (S - sm), axis=1)
        / (np.sqrt(np.nansum((G - gm)**2, axis=1) * np.nansum((S - sm)**2, axis=1)) + 1e-9)
    )

    feat["sanlc_2022"] = df["sanlc_2022"].values
    feat["latitude"]   = df["latitude"].values
    feat["longitude"]  = df["longitude"].values

    print(f"  Feature stats:")
    for col in ["sen_slope", "mk_z", "delta_mean", "cv_gpp", "coupling_corr"]:
        v = feat[col]
        print(f"    {col:<20} mean={v.mean():>7.3f}  std={v.std():>6.3f}  "
              f"min={v.min():>7.3f}  max={v.max():>7.3f}")

    # ── 4. Option A: Rule-based classification ───────────────────────────────
    print_section("OPTION A: Rule-Based Classification")

    sig_pos = (feat["mk_p"] < 0.05) & (feat["sen_slope"] > 0)
    sig_neg = (feat["mk_p"] < 0.05) & (feat["sen_slope"] < 0)
    stable  = (~sig_pos) & (~sig_neg) & (feat["cv_gpp"] < 0.3)
    fluct   = (~sig_pos) & (~sig_neg) & (feat["cv_gpp"] >= 0.3)

    feat["class_A"] = "Fluctuating"
    feat.loc[stable,  "class_A"] = "Stable"
    feat.loc[sig_pos, "class_A"] = "Recovery"
    feat.loc[sig_neg, "class_A"] = "Degradation"

    print("\nClass distribution:")
    vc = feat["class_A"].value_counts()
    for cls, cnt in vc.items():
        print(f"  {cls:<20} {cnt:>8,}  ({cnt/len(feat)*100:.1f}%)")

    print("\nMean trend features per class:")
    cols_show = ["sen_slope", "mk_z", "delta_mean", "cv_gpp", "autocorr_lag1"]
    print(feat.groupby("class_A")[cols_show].mean().round(3).to_string())

    lc_crosstab(feat, "class_A", "Option A")

    # ── 5. Option C: KMeans on trend features ────────────────────────────────
    print_section("OPTION C: KMeans on Trend Features")

    feat_cols_C = ["sen_slope", "mk_z", "early_mean", "late_mean", "delta_mean",
                   "cv_gpp", "autocorr_lag1", "coupling_corr"]
    X = feat[feat_cols_C].fillna(0).values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    for k in [5, 8, 12]:
        print(f"\n  --- KMeans k={k} ---")
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        feat[f"class_C{k}"] = km.fit_predict(X_sc)

        centroids = pd.DataFrame(
            scaler.inverse_transform(km.cluster_centers_),
            columns=feat_cols_C
        ).sort_values("sen_slope")
        print("  Cluster centroids (sorted by sen_slope):")
        print(centroids[["sen_slope", "delta_mean", "cv_gpp", "coupling_corr"]].round(3).to_string())

        lc_crosstab(feat, f"class_C{k}", f"Option C k={k}")

    # ── 6. Current approach comparison (from trajectory_results_v2) ──────────
    print_section("CURRENT APPROACH: catch22 + UMAP + HDBSCAN")

    curr = con.execute(f"""
        SELECT r.cluster, s.sanlc_2022,
               CAST(r.latitude AS FLOAT) as lat, CAST(r.longitude AS FLOAT) as lon
        FROM '{RESULTS}' r
        JOIN '{RAW}' s
          ON CAST(s.latitude  AS FLOAT) = CAST(r.latitude  AS FLOAT)
         AND CAST(s.longitude AS FLOAT) = CAST(r.longitude AS FLOAT)
        USING SAMPLE reservoir(500000 ROWS) REPEATABLE(42)
    """).df()
    curr["lc_name"] = curr["sanlc_2022"].map(SANLC_LABELS).fillna("Unknown")

    print(f"\nCluster distribution ({len(curr):,} sample):")
    vc2 = curr["cluster"].value_counts().sort_index()
    for cls, cnt in vc2.items():
        print(f"  Cluster {cls:<5} {cnt:>8,}  ({cnt/len(curr)*100:.1f}%)")

    purities = []
    for cls, grp in curr.groupby("cluster"):
        lc_cnt = grp["lc_name"].value_counts(normalize=True)
        purities.append(lc_cnt.iloc[0])
    print(f"\n  Avg dominant-LC purity: {np.mean(purities)*100:.1f}%")

    # ── 7. Summary ───────────────────────────────────────────────────────────
    print_section("SUMMARY: Avg Dominant-LC Purity (lower = better LC separation)")

    def avg_purity(df_, col):
        p = []
        for _, grp in df_.groupby(col):
            lc = grp["lc_name"].value_counts(normalize=True)
            p.append(lc.iloc[0] * 100)
        return np.mean(p)

    feat["lc_name"] = feat["sanlc_2022"].map(SANLC_LABELS).fillna("Unknown")
    print(f"  Current (catch22 UMAP):  {np.mean(purities)*100:.1f}%")
    print(f"  Option A (rule-based):   {avg_purity(feat, 'class_A'):.1f}%")
    for k in [5, 8, 12]:
        print(f"  Option C k={k}:           {avg_purity(feat, f'class_C{k}'):.1f}%")

    print("\nDone.")
    con.close()


if __name__ == "__main__":
    main()
