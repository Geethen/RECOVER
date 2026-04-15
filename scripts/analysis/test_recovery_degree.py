"""
Test recovery degree estimation on a single recovering abandoned-ag pixel.

Data flow:
  - Test pixel: abandoned_ag_gpp_2000_2022_SA.parquet (recovering, eco_id=81)
  - Natural reference: dfsubsetNatural.parquet joined to indices_gpp_svh + extracted_indices
    for GPP/SVH time series and 64D embeddings at the same points
  - Transformed reference: indices_gpp_svh_2000_2022.parquet (sanlc NOT IN 1,2)
  - Test embedding: extracted from GEE (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, 2022)

Metrics:
  A.       Percentile within ecoregion natural distribution
  B.       K-nearest natural neighbors ratio
  C (eco). Cosine similarity on AlphaEarth 64D embeddings (ecoregion scope)
  C (local). Cosine similarity on AlphaEarth 64D embeddings (local KNN scope)

Output:
  plots/recovery_degree_comparison.png
  plots/recovery_degree_diagnostic.png
"""
import sys
import ee
import time
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from scipy import stats
from scipy.stats import percentileofscore
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

# ── paths & constants ──────────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
ABANDONED_AG = BASE_DIR / "data" / "abandoned_ag_gpp_2000_2022_SA.parquet"
INDICES_GPP = BASE_DIR / "data" / "indices_gpp_svh_2000_2022.parquet"
EXTRACTED = BASE_DIR / "data" / "extracted_indices.parquet"
NAT_SUBSET = BASE_DIR / "data" / "dfsubsetNatural.parquet"
ECO_CSV = BASE_DIR / "data" / "ecoregion_sds.csv"
OUT_DIR = BASE_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

TARGET_ECO = 81  # Highveld grasslands
YEARS = list(range(2000, 2023))
GPP_COLS = [f"GPP_{y}" for y in YEARS]
SVH_COLS = [f"SVH_{y}" for y in YEARS]
LATE_YEARS = list(range(2018, 2023))
LATE_GPP = [f"GPP_{y}" for y in LATE_YEARS]
LATE_SVH = [f"SVH_{y}" for y in LATE_YEARS]
EMBED_COLS = [f"A{i:02d}" for i in range(64)]
SVH_SCALE = 0.1  # raw SVH values must be multiplied by 0.1
KNN_K = 10
KNN_RADIUS_KM = 50.0
EARTH_R_KM = 6371.0


def load_eco_names():
    df = pd.read_csv(ECO_CSV, usecols=["ECO_ID", "ECO_NAME"]).drop_duplicates("ECO_ID")
    return dict(zip(df["ECO_ID"].astype(int), df["ECO_NAME"]))


def mk_sen_batch(y_arr):
    """Mann-Kendall + Theil-Sen for batch (B, T)."""
    B, T = y_arr.shape
    i, j = np.triu_indices(T, k=1)
    dx = (j - i).astype(np.float32)
    diffs = y_arr[:, j] - y_arr[:, i]
    slopes = diffs / dx[None, :]
    sen_slope = np.median(slopes, axis=1).astype(np.float32)
    S = np.sum(np.sign(diffs), axis=1).astype(np.float64)
    var_s = (T * (T - 1) * (2 * T + 5)) / 18.0
    sigma = np.sqrt(var_s)
    Z = np.zeros(B)
    pos = S > 0; neg = S < 0
    Z[pos] = (S[pos] - 1) / sigma
    Z[neg] = (S[neg] + 1) / sigma
    p = 2 * stats.norm.sf(np.abs(Z))
    return sen_slope, Z, p


# ── Step 1: select test pixel from abandoned_ag ──────────────────────
def select_test_pixel():
    """Find a recovering abandoned-ag pixel in eco_id=81."""
    print("Step 1: Selecting recovering test pixel from abandoned_ag ...")
    gpp_sel = ", ".join(GPP_COLS)
    svh_sel = ", ".join(SVH_COLS)

    sql = f"""
        SELECT pixel_id, latitude, longitude, eco_id, sanlc_2022,
               {gpp_sel}, {svh_sel}
        FROM '{ABANDONED_AG}'
        WHERE eco_id = {TARGET_ECO}
        USING SAMPLE reservoir(80000 ROWS) REPEATABLE(42)
    """
    df = duckdb.sql(sql).df()
    print(f"  Sampled {len(df):,} abandoned-ag pixels (eco_id={TARGET_ECO})")

    gpp = df[GPP_COLS].values.astype(np.float32)
    svh = df[SVH_COLS].values.astype(np.float32) * SVH_SCALE

    gpp_slope, _, gpp_p = mk_sen_batch(gpp)
    svh_slope, _, svh_p = mk_sen_batch(svh)

    # Both significant positive trend
    mask = (gpp_p < 0.05) & (gpp_slope > 0) & (svh_p < 0.05) & (svh_slope > 0)
    print(f"  {mask.sum():,} pixels with both GPP+SVH recovering")

    df_rec = df[mask].copy()
    df_rec["gpp_slope"] = gpp_slope[mask]
    df_rec["svh_slope"] = svh_slope[mask]
    df_rec["gpp_p"] = gpp_p[mask]
    df_rec["svh_p"] = svh_p[mask]

    if len(df_rec) == 0:
        raise RuntimeError("No recovering pixels found — try larger sample")

    # Pick moderate slope (~70th percentile, not extreme outlier)
    df_rec["gpp_rank"] = df_rec["gpp_slope"].rank(pct=True)
    df_rec["svh_rank"] = df_rec["svh_slope"].rank(pct=True)
    df_rec["target_dist"] = abs(df_rec["gpp_rank"] - 0.7) + abs(df_rec["svh_rank"] - 0.7)
    best = df_rec.sort_values("target_dist").iloc[0]
    print(f"  Selected pixel_id={best['pixel_id']}, sanlc_2022={int(best['sanlc_2022'])}")
    print(f"  GPP slope={best['gpp_slope']:.4f} (p={best['gpp_p']:.4f})")
    print(f"  SVH slope={best['svh_slope']:.4f} (p={best['svh_p']:.4f})")
    return best


# ── Step 2: load reference data ──────────────────────────────────────
def load_reference(eco_id, n_natural=10000, n_transformed=5000):
    """Load natural reference from dfsubsetNatural joined to gpp_svh + embeddings.

    Natural pixels are defined by a composite mask applied during GEE extraction:
      (SBTN Natural Lands OR Natural Forests prob >= 0.52)
      AND Global Human Modification <= 0.1
      AND Biodiversity Intactness Index >= 0.7

    Returns nat_df with GPP/SVH + embeddings, and trans_df with GPP/SVH only.
    """
    print(f"\nStep 2: Loading reference data (eco_id={eco_id}) ...")
    gpp_sel = ", ".join(f"g.{c}" for c in GPP_COLS)
    svh_sel = ", ".join(f"g.{c}" for c in SVH_COLS)
    embed_sel = ", ".join(f"n.{c}" for c in EMBED_COLS)

    # Natural reference: dfsubsetNatural joined to indices_gpp_svh
    sql_nat = f"""
        SELECT g.latitude, g.longitude,
               {gpp_sel}, {svh_sel}, {embed_sel}
        FROM '{NAT_SUBSET}' n
        JOIN '{INDICES_GPP}' g
          ON n.id = split_part(g.pixel_id, '_', 1)
        WHERE g.eco_id = {int(eco_id)}
        USING SAMPLE reservoir({n_natural} ROWS) REPEATABLE(42)
    """
    nat_df = duckdb.sql(sql_nat).df()
    print(f"  Natural reference: {len(nat_df):,} pixels (dfsubsetNatural x indices_gpp_svh)")

    # Transformed reference (non-natural sanlc, no embeddings needed for GPP/SVH metrics)
    sql_trans = f"""
        SELECT latitude, longitude,
               {", ".join(GPP_COLS)}, {", ".join(SVH_COLS)}
        FROM '{INDICES_GPP}'
        WHERE sanlc_2022 NOT IN (1, 2) AND eco_id = {int(eco_id)}
        USING SAMPLE reservoir({n_transformed} ROWS) REPEATABLE(42)
    """
    trans_df = duckdb.sql(sql_trans).df()
    print(f"  Transformed reference: {len(trans_df):,} pixels")

    return nat_df, trans_df


def find_knn_natural(test_lat, test_lon, nat_df, k=KNN_K, radius_km=KNN_RADIUS_KM):
    """Find K nearest natural neighbors within radius using BallTree."""
    coords_rad = np.radians(nat_df[["latitude", "longitude"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    test_rad = np.radians([[test_lat, test_lon]])
    radius_rad = radius_km / EARTH_R_KM

    idx = tree.query_radius(test_rad, r=radius_rad)[0]
    if len(idx) < k:
        print(f"  Only {len(idx)} within {radius_km}km, expanding to nearest {k}")
        _, idx = tree.query(test_rad, k=min(k, len(nat_df)))
        idx = idx[0]
    else:
        dists = tree.query(test_rad, k=len(idx))[0][0]
        order = np.argsort(dists)[:k]
        idx = idx[order] if len(idx) > k else idx[:k]

    knn_df = nat_df.iloc[idx].copy()
    print(f"  Found {len(knn_df)} nearest natural neighbors")
    return knn_df


# ── Step 3: extract test pixel embedding from GEE ────────────────────
def extract_embedding_from_gee(lat, lon, project='ee-gsingh'):
    """Extract AlphaEarth 64D embedding from GEE for a single point."""
    print("\nStep 3: Extracting AlphaEarth embedding from GEE ...")
    try:
        ee.Initialize(project=project,
                      opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception:
        ee.Initialize(project=project)
    print(f"  [OK] GEE initialized (project={project})")

    # AlphaEarth foundation model embeddings (2022 annual mosaic)
    aef = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    aef_2022 = aef.filterDate("2022-01-01", "2023-01-01").mosaic()

    point = ee.Geometry.Point([lon, lat])
    sample = aef_2022.sample(region=point, scale=10, numPixels=1, geometries=False)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = sample.getInfo()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt+1}/{max_retries} after {wait}s: {str(e)[:80]}")
                time.sleep(wait)
            else:
                raise

    features = result.get("features", [])
    if not features:
        raise RuntimeError(f"No embedding returned for ({lat}, {lon})")

    props = features[0]["properties"]
    embedding = np.zeros(64)
    for i in range(64):
        for key in [f"A{i:02d}", f"b{i}", f"B{i}", f"{i}", f"embedding_{i}"]:
            if key in props:
                embedding[i] = props[key]
                break

    # If A00-style keys not found, map by sorted band keys
    if embedding.sum() == 0:
        band_keys = sorted([k for k in props.keys() if k not in ('system:index',)])
        if len(band_keys) >= 64:
            for i, key in enumerate(band_keys[:64]):
                embedding[i] = props[key]
            print(f"  Mapped {len(band_keys)} bands to embedding (keys: {band_keys[:3]}...)")
        else:
            raise RuntimeError(f"Expected 64 bands, got {len(band_keys)}: {band_keys}")

    print(f"  [OK] Extracted 64D embedding at ({lat:.4f}, {lon:.4f})")
    print(f"  Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    return embedding


# ── Step 4: metrics ───────────────────────────────────────────────────
def metric_a_percentile(test_pixel, nat_df, trans_df):
    """Percentile rank within ecoregion natural distribution."""
    nat_gpp_means = nat_df[LATE_GPP].values.astype(float).mean(axis=1)
    nat_svh_means = nat_df[LATE_SVH].values.astype(float).mean(axis=1)
    test_gpp = test_pixel[LATE_GPP].values.astype(float).mean()
    test_svh = test_pixel[LATE_SVH].values.astype(float).mean()
    gpp_pctl = percentileofscore(nat_gpp_means, test_gpp)
    svh_pctl = percentileofscore(nat_svh_means, test_svh)

    # Also compute where transformed pixels fall
    trans_gpp_means = trans_df[LATE_GPP].values.astype(float).mean(axis=1)
    trans_svh_means = trans_df[LATE_SVH].values.astype(float).mean(axis=1)
    trans_gpp_pctl = np.mean([percentileofscore(nat_gpp_means, v) for v in trans_gpp_means])
    trans_svh_pctl = np.mean([percentileofscore(nat_svh_means, v) for v in trans_svh_means])

    return {
        "GPP": gpp_pctl, "SVH": svh_pctl,
        "GPP_pctl": gpp_pctl, "SVH_pctl": svh_pctl,
        "trans_GPP_pctl": trans_gpp_pctl, "trans_SVH_pctl": trans_svh_pctl,
        "desc": "percentile (50th = median natural)",
    }


def metric_b_knn(test_pixel, knn_df, nat_df):
    """Ratio to K-nearest natural neighbors mean, percentile-normalized."""
    test_gpp = test_pixel[LATE_GPP].values.astype(float).mean()
    test_svh = test_pixel[LATE_SVH].values.astype(float).mean()
    knn_gpp = knn_df[LATE_GPP].values.astype(float).mean()
    knn_svh = knn_df[LATE_SVH].values.astype(float).mean()
    gpp_ratio = test_gpp / (knn_gpp + 1e-9)
    svh_ratio = test_svh / (knn_svh + 1e-9)
    # Natural distribution: each natural pixel's ratio vs eco mean
    nat_gpp_means = nat_df[LATE_GPP].values.astype(float).mean(axis=1)
    nat_svh_means = nat_df[LATE_SVH].values.astype(float).mean(axis=1)
    eco_gpp_mean = nat_gpp_means.mean()
    eco_svh_mean = nat_svh_means.mean()
    nat_gpp_ratios = nat_gpp_means / (eco_gpp_mean + 1e-9)
    nat_svh_ratios = nat_svh_means / (eco_svh_mean + 1e-9)
    return {
        "GPP": gpp_ratio, "SVH": svh_ratio,
        "GPP_pctl": percentileofscore(nat_gpp_ratios, gpp_ratio),
        "SVH_pctl": percentileofscore(nat_svh_ratios, svh_ratio),
        "desc": f"KNN ratio (K={len(knn_df)}, 1.0 = at local natural)",
    }


def metric_c_cosine_eco(test_embedding, nat_df):
    """Cosine similarity of test embedding vs ecoregion natural embeddings."""
    nat_embeds = nat_df[EMBED_COLS].values.astype(float)
    test_embed = test_embedding.reshape(1, -1)

    # Test vs all ecoregion natural
    cos_sims = cosine_similarity(test_embed, nat_embeds)[0]
    mean_sim = cos_sims.mean()

    # Baseline: natural-natural mean cosine sims (sample up to 500)
    rng = np.random.RandomState(42)
    n = len(nat_embeds)
    sample_n = min(500, n)
    sample_idx = rng.choice(n, sample_n, replace=False)
    nat_mean_sims = cosine_similarity(nat_embeds[sample_idx], nat_embeds).mean(axis=1)

    pctl = percentileofscore(nat_mean_sims, mean_sim)
    return {
        "GPP": mean_sim, "SVH": mean_sim,
        "GPP_pctl": pctl, "SVH_pctl": pctl,
        "desc": "cosine sim (ecoregion)",
        "nat_nat_mean": nat_mean_sims.mean(),
        "nat_nat_std": nat_mean_sims.std(),
    }


def metric_c_cosine_local(test_embedding, knn_df, nat_df, k=KNN_K, n_baseline=300):
    """Cosine similarity to KNN, percentile-normalised against eco natural baseline.

    Same pattern as Metric B: compute test's mean cosine sim to its K nearest
    natural neighbors, then build a smooth baseline by sampling n_baseline natural
    pixels and computing each one's mean cosine sim to *its own* KNN.
    """
    knn_embeds = knn_df[EMBED_COLS].values.astype(float)
    test_embed = test_embedding.reshape(1, -1)

    # Test vs KNN natural
    cos_sims = cosine_similarity(test_embed, knn_embeds)[0]
    mean_sim = cos_sims.mean()

    # Baseline: sample natural pixels, each computes cosine sim to its own KNN
    nat_embeds = nat_df[EMBED_COLS].values.astype(float)
    nat_coords = np.radians(nat_df[["latitude", "longitude"]].values)
    tree = BallTree(nat_coords, metric="haversine")

    rng = np.random.RandomState(42)
    n = len(nat_embeds)
    sample_n = min(n_baseline, n)
    sample_idx = rng.choice(n, sample_n, replace=False)

    baseline_sims = []
    for idx in sample_idx:
        # Find this natural pixel's K nearest neighbors (excluding itself)
        _, nn_idx = tree.query(nat_coords[idx:idx+1], k=k + 1)
        nn_idx = nn_idx[0]
        nn_idx = nn_idx[nn_idx != idx][:k]  # exclude self
        nn_embeds = nat_embeds[nn_idx]
        s = cosine_similarity(nat_embeds[idx:idx+1], nn_embeds)[0].mean()
        baseline_sims.append(s)
    baseline_sims = np.array(baseline_sims)

    pctl = percentileofscore(baseline_sims, mean_sim)
    return {
        "GPP": mean_sim, "SVH": mean_sim,
        "GPP_pctl": pctl, "SVH_pctl": pctl,
        "desc": f"cosine sim (local K={k})",
        "baseline_mean": baseline_sims.mean(),
        "baseline_std": baseline_sims.std(),
    }


# ── Aggregation ────────────────────────────────────────────────────────
def aggregate_percentiles(results):
    """Average all approach percentiles into a composite recovery score (0-100)."""
    gpp_pctls, svh_pctls = [], []
    for key in sorted(results):
        r = results[key]
        if r["GPP_pctl"] is not None:
            gpp_pctls.append(r["GPP_pctl"])
        if r.get("SVH_pctl") is not None:
            svh_pctls.append(r["SVH_pctl"])
    composite_gpp = np.mean(gpp_pctls) if gpp_pctls else None
    composite_svh = np.mean(svh_pctls) if svh_pctls else None
    vals = [v for v in [composite_gpp, composite_svh] if v is not None]
    composite_overall = np.mean(vals) if vals else None
    return composite_gpp, composite_svh, composite_overall


def gradient_position(test_pctl, trans_pctl):
    """Position test pixel on transformed-to-natural gradient (0=transformed, 1=natural)."""
    if trans_pctl >= 50:
        return None
    return (test_pctl - trans_pctl) / (50 - trans_pctl) if (50 - trans_pctl) > 0 else None


# ── Visualisation ──────────────────────────────────────────────────────
def create_figure(test_pixel, nat_df, trans_df, knn_df, results, eco_name, composites):
    composite_gpp, composite_svh, composite_overall = composites
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Recovery Degree Estimation — pixel_id={test_pixel['pixel_id']}\n"
        f"Ecoregion: {eco_name}  |  sanlc_2022={int(test_pixel['sanlc_2022'])}  |  "
        f"lat={test_pixel['latitude']:.3f}, lon={test_pixel['longitude']:.3f}",
        fontsize=12, fontweight="bold",
    )

    nat_gpp = nat_df[GPP_COLS].values.astype(float)
    nat_svh = nat_df[SVH_COLS].values.astype(float) * SVH_SCALE
    trans_gpp = trans_df[GPP_COLS].values.astype(float)
    trans_svh = trans_df[SVH_COLS].values.astype(float) * SVH_SCALE
    knn_gpp_mean = knn_df[GPP_COLS].values.astype(float).mean(axis=0)
    knn_svh_mean = knn_df[SVH_COLS].values.astype(float).mean(axis=0) * SVH_SCALE
    test_gpp = test_pixel[GPP_COLS].values.astype(float)
    test_svh = test_pixel[SVH_COLS].values.astype(float) * SVH_SCALE

    # Panel (0,0): GPP trajectory
    ax = axes[0, 0]
    ax.fill_between(YEARS, nat_gpp.mean(0) - nat_gpp.std(0), nat_gpp.mean(0) + nat_gpp.std(0),
                    alpha=0.2, color="#2ecc71", label="Natural ref +/-1s")
    ax.plot(YEARS, nat_gpp.mean(0), "--", color="#2ecc71", lw=1.5, label="Natural mean")
    ax.plot(YEARS, trans_gpp.mean(0), "--", color="#e67e22", lw=1.5, label="Transformed mean")
    ax.plot(YEARS, knn_gpp_mean, "--", color="#3498db", lw=1.5, label=f"KNN mean (K={len(knn_df)})")
    ax.plot(YEARS, test_gpp, "-", color="#e74c3c", lw=2.5, label="Test pixel")
    ax.set_title("GPP trajectory"); ax.set_xlabel("Year"); ax.set_ylabel("GPP")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel (0,1): SVH trajectory
    ax = axes[0, 1]
    ax.fill_between(YEARS, nat_svh.mean(0) - nat_svh.std(0), nat_svh.mean(0) + nat_svh.std(0),
                    alpha=0.2, color="#2ecc71", label="Natural ref +/-1s")
    ax.plot(YEARS, nat_svh.mean(0), "--", color="#2ecc71", lw=1.5, label="Natural mean")
    ax.plot(YEARS, trans_svh.mean(0), "--", color="#e67e22", lw=1.5, label="Transformed mean")
    ax.plot(YEARS, knn_svh_mean, "--", color="#3498db", lw=1.5, label=f"KNN mean (K={len(knn_df)})")
    ax.plot(YEARS, test_svh, "-", color="#e74c3c", lw=2.5, label="Test pixel")
    ax.set_title("SVH trajectory"); ax.set_xlabel("Year"); ax.set_ylabel("SVH")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel (1,0): Percentile bar chart
    ax = axes[1, 0]
    keys = sorted(results)
    labels_short = {
        "A": "A.Pctile", "B": "B.KNN",
        "C_eco": "C.Cos(eco)", "C_local": "C.Cos(local)",
    }
    xlabels = [labels_short.get(k, k) for k in keys]
    gpp_pctls = [results[k]["GPP_pctl"] for k in keys]
    svh_pctls = [results[k].get("SVH_pctl") for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w/2, gpp_pctls, w, label="GPP", color="#e74c3c", alpha=0.7)
    svh_vals = [v if v is not None else 0 for v in svh_pctls]
    ax.bar(x + w/2, svh_vals, w, label="SVH", color="#3498db", alpha=0.7)
    ax.axhline(50, color="gray", ls=":", lw=1, label="50th (median natural)")
    # Show transformed baseline for metric A
    if "A" in results and "trans_GPP_pctl" in results["A"]:
        ai = keys.index("A")
        ax.scatter(ai - w/2, results["A"]["trans_GPP_pctl"], marker="v", color="#e67e22",
                   s=60, zorder=5, label="Transformed mean")
        ax.scatter(ai + w/2, results["A"]["trans_SVH_pctl"], marker="v", color="#e67e22", s=60, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8, rotation=30)
    ax.set_ylabel("Percentile within natural distribution")
    ax.set_ylim(0, 105)
    ax.set_title("Percentile-normalized recovery scores")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")

    # Panel (1,1): Text summary
    ax = axes[1, 1]
    ax.axis("off")
    lines = [f"{'Metric':<30} {'Raw GPP':>9} {'Raw SVH':>9} {'Pctl GPP':>9} {'Pctl SVH':>9}"]
    lines.append("-" * 72)
    for key in keys:
        r = results[key]
        gpp_s = f"{r['GPP']:.3f}" if r["GPP"] is not None else "n/a"
        svh_s = f"{r['SVH']:.3f}" if r["SVH"] is not None else "n/a"
        gp = f"{r['GPP_pctl']:.1f}" if r["GPP_pctl"] is not None else "n/a"
        sp = f"{r['SVH_pctl']:.1f}" if r.get("SVH_pctl") is not None else "n/a"
        lines.append(f"{key}. {r['desc']:<27} {gpp_s:>9} {svh_s:>9} {gp:>9} {sp:>9}")
    lines.append("-" * 72)
    cg = f"{composite_gpp:.1f}" if composite_gpp is not None else "n/a"
    cs = f"{composite_svh:.1f}" if composite_svh is not None else "n/a"
    co = f"{composite_overall:.1f}" if composite_overall is not None else "n/a"
    lines.append(f"{'COMPOSITE (mean pctl)':<48} {cg:>9} {cs:>9}")
    lines.append(f"{'OVERALL RECOVERY SCORE':<48} {co:>9}")
    # Cosine baselines
    for ck in ["C_eco", "C_local"]:
        if ck in results:
            rc = results[ck]
            if "nat_nat_mean" in rc:
                lines.append(f"\n{ck} nat-nat baseline: {rc['nat_nat_mean']:.3f} +/- {rc['nat_nat_std']:.3f}")
            if "baseline_mean" in rc:
                lines.append(f"\n{ck} nat KNN baseline: {rc['baseline_mean']:.3f} +/- {rc['baseline_std']:.3f}")
    if "A" in results and "trans_GPP_pctl" in results["A"]:
        rc = results["A"]
        gpp_grad = gradient_position(rc["GPP_pctl"], rc["trans_GPP_pctl"])
        svh_grad = gradient_position(rc["SVH_pctl"], rc["trans_SVH_pctl"])
        lines.append("\nGradient position (0=transformed, 1=natural):")
        lines.append(f"  GPP: {gpp_grad:.2f}" if gpp_grad is not None else "  GPP: n/a")
        lines.append(f"  SVH: {svh_grad:.2f}" if svh_grad is not None else "  SVH: n/a")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=7, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out_path = OUT_DIR / "recovery_degree_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {out_path}")


def create_diagnostic(test_pixel, nat_df, trans_df, knn_df, eco_name):
    """Raw-data diagnostic: distributions, spatial map, time series."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Diagnostic — pixel_id={test_pixel['pixel_id']}  |  "
        f"{eco_name} (eco_id={int(test_pixel['eco_id'])})\n"
        f"sanlc_2022={int(test_pixel['sanlc_2022'])}  |  "
        f"lat={test_pixel['latitude']:.4f}, lon={test_pixel['longitude']:.4f}\n"
        f"Natural ref: dfsubsetNatural (composite mask) x indices_gpp_svh",
        fontsize=10, fontweight="bold",
    )

    nat_gpp_late = nat_df[LATE_GPP].values.astype(float).mean(axis=1)
    nat_svh_late = nat_df[LATE_SVH].values.astype(float).mean(axis=1) * SVH_SCALE
    trans_gpp_late = trans_df[LATE_GPP].values.astype(float).mean(axis=1)
    trans_svh_late = trans_df[LATE_SVH].values.astype(float).mean(axis=1) * SVH_SCALE
    test_gpp_late = test_pixel[LATE_GPP].values.astype(float).mean()
    test_svh_late = test_pixel[LATE_SVH].values.astype(float).mean() * SVH_SCALE

    nat_gpp_ts = nat_df[GPP_COLS].values.astype(float)
    nat_svh_ts = nat_df[SVH_COLS].values.astype(float) * SVH_SCALE
    knn_gpp_ts = knn_df[GPP_COLS].values.astype(float)
    knn_svh_ts = knn_df[SVH_COLS].values.astype(float) * SVH_SCALE
    test_gpp_ts = test_pixel[GPP_COLS].values.astype(float)
    test_svh_ts = test_pixel[SVH_COLS].values.astype(float) * SVH_SCALE

    # (0,0) GPP late-period histogram
    ax = axes[0, 0]
    ax.hist(nat_gpp_late, bins=50, color="#2ecc71", alpha=0.5, label=f"Natural (n={len(nat_df)})")
    ax.hist(trans_gpp_late, bins=50, color="#e67e22", alpha=0.4, label=f"Transformed (n={len(trans_df)})")
    ax.axvline(test_gpp_late, color="#e74c3c", lw=2.5, label=f"Test = {test_gpp_late:.0f}")
    ax.axvline(np.median(nat_gpp_late), color="#2ecc71", ls="--", lw=1.5, label=f"Nat median = {np.median(nat_gpp_late):.0f}")
    ax.set_title("GPP late-period mean (2018-2022)")
    ax.set_xlabel("GPP"); ax.set_ylabel("Count")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (0,1) SVH late-period histogram
    ax = axes[0, 1]
    ax.hist(nat_svh_late, bins=50, color="#2ecc71", alpha=0.5, label=f"Natural (n={len(nat_df)})")
    ax.hist(trans_svh_late, bins=50, color="#e67e22", alpha=0.4, label=f"Transformed (n={len(trans_df)})")
    ax.axvline(test_svh_late, color="#e74c3c", lw=2.5, label=f"Test = {test_svh_late:.1f}")
    ax.axvline(np.median(nat_svh_late), color="#2ecc71", ls="--", lw=1.5, label=f"Nat median = {np.median(nat_svh_late):.1f}")
    ax.set_title("SVH late-period mean (2018-2022)")
    ax.set_xlabel("SVH"); ax.set_ylabel("Count")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (0,2) Spatial map
    ax = axes[0, 2]
    ax.scatter(nat_df["longitude"], nat_df["latitude"], s=1, c="#2ecc71", alpha=0.3, label="Natural ref")
    ax.scatter(trans_df["longitude"], trans_df["latitude"], s=1, c="#e67e22", alpha=0.2, label="Transformed ref")
    ax.scatter(knn_df["longitude"], knn_df["latitude"], s=40, c="#3498db",
               edgecolors="k", lw=0.5, zorder=5, label=f"KNN (K={len(knn_df)})")
    ax.scatter(test_pixel["longitude"], test_pixel["latitude"], s=120, c="#e74c3c",
               marker="*", edgecolors="k", lw=0.8, zorder=10, label="Test pixel")
    theta = np.linspace(0, 2*np.pi, 100)
    r_deg = KNN_RADIUS_KM / 111.0
    ax.plot(test_pixel["longitude"] + r_deg * np.cos(theta),
            test_pixel["latitude"] + r_deg * np.sin(theta),
            ":", color="gray", lw=1, label=f"{KNN_RADIUS_KM:.0f}km radius")
    ax.set_title("Spatial distribution")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(fontsize=6, loc="upper left"); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # (1,0) GPP spaghetti
    ax = axes[1, 0]
    rng = np.random.RandomState(42)
    n_show = min(100, len(nat_gpp_ts))
    idx = rng.choice(len(nat_gpp_ts), n_show, replace=False)
    for i in idx:
        ax.plot(YEARS, nat_gpp_ts[i], color="#2ecc71", alpha=0.08, lw=0.5)
    ax.plot(YEARS, nat_gpp_ts.mean(0), "--", color="#2ecc71", lw=2, label="Natural mean")
    ax.plot(YEARS, trans_df[GPP_COLS].values.astype(float).mean(0), "--", color="#e67e22", lw=2, label="Transformed mean")
    for i in range(len(knn_gpp_ts)):
        ax.plot(YEARS, knn_gpp_ts[i], color="#3498db", alpha=0.4, lw=0.8)
    ax.plot(YEARS, knn_gpp_ts.mean(0), "--", color="#3498db", lw=2, label="KNN mean")
    ax.plot(YEARS, test_gpp_ts, "-", color="#e74c3c", lw=2.5, label="Test pixel")
    ax.set_title("GPP time series"); ax.set_xlabel("Year"); ax.set_ylabel("GPP")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (1,1) SVH spaghetti
    ax = axes[1, 1]
    for i in idx:
        ax.plot(YEARS, nat_svh_ts[i], color="#2ecc71", alpha=0.08, lw=0.5)
    ax.plot(YEARS, nat_svh_ts.mean(0), "--", color="#2ecc71", lw=2, label="Natural mean")
    ax.plot(YEARS, trans_df[SVH_COLS].values.astype(float).mean(0), "--", color="#e67e22", lw=2, label="Transformed mean")
    for i in range(len(knn_svh_ts)):
        ax.plot(YEARS, knn_svh_ts[i], color="#3498db", alpha=0.4, lw=0.8)
    ax.plot(YEARS, knn_svh_ts.mean(0), "--", color="#3498db", lw=2, label="KNN mean")
    ax.plot(YEARS, test_svh_ts, "-", color="#e74c3c", lw=2.5, label="Test pixel")
    ax.set_title("SVH time series"); ax.set_xlabel("Year"); ax.set_ylabel("SVH")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (1,2) GPP vs SVH scatter
    ax = axes[1, 2]
    ax.scatter(nat_gpp_late, nat_svh_late, s=3, c="#2ecc71", alpha=0.3, label="Natural")
    ax.scatter(trans_gpp_late, trans_svh_late, s=3, c="#e67e22", alpha=0.2, label="Transformed")
    knn_gpp_late = knn_df[LATE_GPP].values.astype(float).mean(axis=1)
    knn_svh_late = knn_df[LATE_SVH].values.astype(float).mean(axis=1) * SVH_SCALE
    ax.scatter(knn_gpp_late, knn_svh_late, s=40, c="#3498db",
               edgecolors="k", lw=0.5, zorder=5, label="KNN")
    ax.scatter(test_gpp_late, test_svh_late, s=120, c="#e74c3c",
               marker="*", edgecolors="k", lw=0.8, zorder=10, label="Test pixel")
    ax.set_title("GPP vs SVH (late-period means)")
    ax.set_xlabel("GPP (2018-2022 mean)"); ax.set_ylabel("SVH (2018-2022 mean)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "recovery_degree_diagnostic.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Diagnostic plot saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────
def main():
    eco_dict = load_eco_names()

    # Step 1: select test pixel from abandoned_ag
    test = select_test_pixel()
    eco_id = int(test["eco_id"])
    eco_name = eco_dict.get(eco_id, f"eco_{eco_id}")
    print(f"  Ecoregion: {eco_name}")

    # Step 2: load reference (natural from dfsubsetNatural join, transformed from gpp_svh)
    nat_df, trans_df = load_reference(eco_id)
    knn_df = find_knn_natural(test["latitude"], test["longitude"], nat_df)

    # Step 3: extract test pixel embedding from GEE
    test_embedding = extract_embedding_from_gee(test["latitude"], test["longitude"])

    # Step 4: compute metrics
    print("\nStep 4: Computing recovery degree ...")
    results = {}
    results["A"] = metric_a_percentile(test, nat_df, trans_df)
    results["B"] = metric_b_knn(test, knn_df, nat_df)
    results["C_eco"] = metric_c_cosine_eco(test_embedding, nat_df)
    results["C_local"] = metric_c_cosine_local(test_embedding, knn_df, nat_df)

    # Aggregate
    composite_gpp, composite_svh, composite_overall = aggregate_percentiles(results)

    # Print summary
    print("\n" + "=" * 80)
    print("  RECOVERY DEGREE ESTIMATION — SUMMARY")
    print("=" * 80)
    print(f"  Test pixel: {test['pixel_id']}  |  sanlc_2022={int(test['sanlc_2022'])}")
    print(f"  Source: abandoned_ag_gpp_2000_2022_SA.parquet")
    print(f"  Ecoregion: {eco_name} (id={eco_id})")
    print(f"  Location: ({test['latitude']:.4f}, {test['longitude']:.4f})")
    print(f"  GPP slope: {test['gpp_slope']:.4f} (p={test['gpp_p']:.4f})")
    print(f"  SVH slope: {test['svh_slope']:.4f} (p={test['svh_p']:.4f})")
    print(f"  Natural ref: dfsubsetNatural x indices_gpp_svh (composite mask)")
    print("-" * 80)
    print(f"  {'Metric':<37} {'GPP':>8} {'SVH':>8} {'Pctl GPP':>9} {'Pctl SVH':>9}")
    print("-" * 80)
    for key in sorted(results):
        r = results[key]
        gpp_s = f"{r['GPP']:.3f}" if r["GPP"] is not None else "n/a"
        svh_s = f"{r['SVH']:.3f}" if r["SVH"] is not None else "n/a"
        gp = f"{r['GPP_pctl']:.1f}" if r["GPP_pctl"] is not None else "n/a"
        sp = f"{r['SVH_pctl']:.1f}" if r.get("SVH_pctl") is not None else "n/a"
        print(f"  {key}. {r['desc']:<34} {gpp_s:>8} {svh_s:>8} {gp:>9} {sp:>9}")
    print("-" * 80)
    cg = f"{composite_gpp:.1f}" if composite_gpp is not None else "n/a"
    cs = f"{composite_svh:.1f}" if composite_svh is not None else "n/a"
    co = f"{composite_overall:.1f}" if composite_overall is not None else "n/a"
    print(f"  {'COMPOSITE (mean percentile)':<57} {cg:>9} {cs:>9}")
    print(f"  {'OVERALL RECOVERY SCORE':<57} {co:>9}")

    # Gradient position
    if "A" in results and "trans_GPP_pctl" in results["A"]:
        rc = results["A"]
        gpp_grad = gradient_position(rc["GPP_pctl"], rc["trans_GPP_pctl"])
        svh_grad = gradient_position(rc["SVH_pctl"], rc["trans_SVH_pctl"])
        print(f"\n  Gradient position (0=transformed, 1=natural):")
        print(f"    GPP: {gpp_grad:.2f}" if gpp_grad is not None else "    GPP: n/a")
        print(f"    SVH: {svh_grad:.2f}" if svh_grad is not None else "    SVH: n/a")
    print("=" * 80)

    # Step 5: visualise
    print("\nStep 5: Creating visualisations ...")
    create_diagnostic(test, nat_df, trans_df, knn_df, eco_name)
    create_figure(test, nat_df, trans_df, knn_df, results, eco_name,
                  (composite_gpp, composite_svh, composite_overall))
    print("Done.")


if __name__ == "__main__":
    main()
