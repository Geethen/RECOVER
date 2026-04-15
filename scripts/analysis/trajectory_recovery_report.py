"""
Trajectory recovery analysis report.

1. Load raw data, drop transformed LC pixels (keep sanlc_2022 in {1,2})
2. Summarize LC percentages by ecoregion
3. Compute Sen's slope + Mann-Kendall for GPP (functional) and SVH (structural)
4. Classify: Recovery / Stable / Degradation for each
5. Cross-tabulate GPP x SVH classes
6. Output markdown report
"""
import sys
import gc
import numpy as np
import pandas as pd
import duckdb
from scipy import stats
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
RAW = BASE_DIR / "data" / "abandoned_ag_gpp_2000_2022_SA.parquet"
REPORT = BASE_DIR / "reports" / "trajectory_recovery_report.md"
YEARS = list(range(2000, 2023))
GPP_COLS = [f"GPP_{y}" for y in YEARS]
SVH_COLS = [f"SVH_{y}" for y in YEARS]

SANLC_LABELS = {
    1: "Natural/near-natural", 2: "Secondary natural", 3: "Artificial water",
    4: "Built-up", 5: "Cropland", 6: "Mine", 7: "Plantation",
}
NATURAL_CODES = [1, 2]
ECO_CSV = BASE_DIR / "data" / "ecoregion_sds.csv"


def load_eco_names():
    """Load ECO_ID -> ECO_NAME mapping from ecoregion CSV."""
    df = pd.read_csv(ECO_CSV, usecols=["ECO_ID", "ECO_NAME"]).drop_duplicates("ECO_ID")
    return dict(zip(df["ECO_ID"].astype(int), df["ECO_NAME"]))


def mk_sen_batch(y_arr):
    """Mann-Kendall + Theil-Sen for small batch (B, T). B should be <= 1000."""
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


def process_chunk(chunk_df, eco_means):
    """Process a chunk: eco-standardize, compute trends, classify, return labels."""
    gpp = chunk_df[GPP_COLS].values.astype(np.float32)
    svh = chunk_df[SVH_COLS].values.astype(np.float32)
    ecos = chunk_df["eco_id"].values

    # Eco-standardize using precomputed global eco means
    for eco in np.unique(ecos):
        m = ecos == eco
        if eco in eco_means:
            em = eco_means[eco]
            gpp[m] = (gpp[m] - em["gm"]) / em["gs"]
            svh[m] = (svh[m] - em["sm"]) / em["ss"]
        else:
            gpp[m] = (gpp[m] - np.nanmean(gpp[m])) / (np.nanstd(gpp[m]) + 1e-9)
            svh[m] = (svh[m] - np.nanmean(svh[m])) / (np.nanstd(svh[m]) + 1e-9)

    # Process in small sub-batches for memory
    B = len(gpp)
    SB = 500
    gpp_slope = np.empty(B, dtype=np.float32)
    gpp_p = np.empty(B, dtype=np.float64)
    svh_slope = np.empty(B, dtype=np.float32)
    svh_p = np.empty(B, dtype=np.float64)

    for s in range(0, B, SB):
        e = min(s + SB, B)
        gs, _, gp = mk_sen_batch(gpp[s:e])
        ss, _, sp = mk_sen_batch(svh[s:e])
        gpp_slope[s:e] = gs
        gpp_p[s:e] = gp
        svh_slope[s:e] = ss
        svh_p[s:e] = sp

    del gpp, svh
    gpp_class = classify_trend(gpp_slope, gpp_p)
    svh_class = classify_trend(svh_slope, svh_p)

    return ecos, gpp_class, svh_class, gpp_slope, svh_slope


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    eco_names = load_eco_names()

    def ename(eco_id):
        return eco_names.get(int(eco_id), f"Ecoregion {eco_id}")

    # ── 1. Metadata queries via DuckDB ──
    print("Loading metadata...")
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=4")

    total_n = con.execute(f"SELECT COUNT(*) FROM '{RAW}'").fetchone()[0]
    print(f"  Total pixels: {total_n:,}")

    lc_dist_all = con.execute(f"""
        SELECT sanlc_2022, COUNT(*) AS n
        FROM '{RAW}' GROUP BY sanlc_2022 ORDER BY sanlc_2022
    """).df()

    lc_by_eco = con.execute(f"""
        SELECT eco_id, sanlc_2022, COUNT(*) AS n
        FROM '{RAW}' GROUP BY eco_id, sanlc_2022 ORDER BY eco_id, sanlc_2022
    """).df()

    filtered_n = con.execute(f"""
        SELECT COUNT(*) FROM '{RAW}' WHERE sanlc_2022 IN (1, 2)
    """).fetchone()[0]
    print(f"  Natural-LC pixels: {filtered_n:,} ({filtered_n/total_n*100:.1f}%)")

    con.close()

    # ── 2. LC by ecoregion pivot ──
    lc_by_eco["lc_name"] = lc_by_eco["sanlc_2022"].map(SANLC_LABELS)
    eco_totals = lc_by_eco.groupby("eco_id")["n"].sum()
    lc_by_eco["pct"] = (lc_by_eco["n"] / lc_by_eco["eco_id"].map(eco_totals) * 100).round(2)
    lc_eco_pivot = lc_by_eco.pivot_table(index="eco_id", columns="lc_name", values="pct", fill_value=0)
    lc_eco_pivot["Total pixels"] = eco_totals

    # ── 3. Process trends in chunks using pyarrow ──
    print("Computing trends for natural-LC pixels...")
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(RAW))
    needed_cols = ["eco_id", "sanlc_2022"] + GPP_COLS + SVH_COLS

    # Accumulators
    gpp_counts = Counter()
    svh_counts = Counter()
    cross_counts = Counter()
    eco_gpp_counts = Counter()
    eco_svh_counts = Counter()
    eco_totals_nat = Counter()
    eco_cross_counts = Counter()
    slope_sums = {c: {"gpp": 0.0, "svh": 0.0, "n": 0} for c in ["Recovery", "Stable", "Degradation"]}

    processed = 0
    for batch in pf.iter_batches(batch_size=20_000, columns=needed_cols):
        chunk_df = batch.to_pandas()
        # Filter to natural LC
        mask = chunk_df["sanlc_2022"].isin(NATURAL_CODES)
        chunk_df = chunk_df[mask].reset_index(drop=True)
        if len(chunk_df) == 0:
            continue

        # Process
        gpp = chunk_df[GPP_COLS].values.astype(np.float32)
        svh = chunk_df[SVH_COLS].values.astype(np.float32)
        ecos = chunk_df["eco_id"].values

        # Within-chunk eco standardization
        for eco in np.unique(ecos):
            m = ecos == eco
            gpp[m] = (gpp[m] - np.nanmean(gpp[m])) / (np.nanstd(gpp[m]) + 1e-9)
            svh[m] = (svh[m] - np.nanmean(svh[m])) / (np.nanstd(svh[m]) + 1e-9)

        # Trend in sub-batches
        B = len(gpp)
        SB = 500
        gpp_slope = np.empty(B, dtype=np.float32)
        gpp_pvals = np.empty(B, dtype=np.float64)
        svh_slope = np.empty(B, dtype=np.float32)
        svh_pvals = np.empty(B, dtype=np.float64)

        for s in range(0, B, SB):
            e = min(s + SB, B)
            gs, _, gp = mk_sen_batch(gpp[s:e])
            ss_, _, sp = mk_sen_batch(svh[s:e])
            gpp_slope[s:e] = gs
            gpp_pvals[s:e] = gp
            svh_slope[s:e] = ss_
            svh_pvals[s:e] = sp

        del gpp, svh
        gpp_class = classify_trend(gpp_slope, gpp_pvals)
        svh_class = classify_trend(svh_slope, svh_pvals)

        # Accumulate using pandas groupby (vectorized)
        agg = pd.DataFrame({"eco": ecos, "gc": gpp_class, "sc": svh_class,
                            "gsl": gpp_slope, "ssl": svh_slope})

        for (gc_, sc_), cnt in agg.groupby(["gc", "sc"]).size().items():
            cross_counts[(gc_, sc_)] += cnt
            gpp_counts[gc_] += cnt

        for sc_ in agg["sc"].unique():
            svh_counts[sc_] += int((agg["sc"] == sc_).sum())

        for (eco_, gc_), cnt in agg.groupby(["eco", "gc"]).size().items():
            eco_gpp_counts[(eco_, gc_)] += cnt

        for (eco_, sc_), cnt in agg.groupby(["eco", "sc"]).size().items():
            eco_svh_counts[(eco_, sc_)] += cnt

        for eco_, cnt in agg.groupby("eco").size().items():
            eco_totals_nat[eco_] += cnt

        for (eco_, gc_, sc_), cnt in agg.groupby(["eco", "gc", "sc"]).size().items():
            eco_cross_counts[(eco_, gc_, sc_)] += cnt

        # Slope sums
        for gc_ in agg["gc"].unique():
            m = agg["gc"] == gc_
            slope_sums[gc_]["gpp"] += float(agg.loc[m, "gsl"].sum())
            slope_sums[gc_]["svh"] += float(agg.loc[m, "ssl"].sum())
            slope_sums[gc_]["n"] += int(m.sum())

        processed += len(chunk_df)
        del chunk_df, agg, gpp_slope, svh_slope, gpp_pvals, svh_pvals, gpp_class, svh_class
        gc.collect()

        if processed % 200_000 < 20_000:
            print(f"  {processed:,} / {filtered_n:,} ({processed/filtered_n*100:.1f}%)")

    del pf
    print(f"  Done. Total: {processed:,}")

    # ── 4. Build tables ──
    classes = ["Recovery", "Stable", "Degradation"]
    N = sum(gpp_counts.values())

    gpp_dist = {c: gpp_counts.get(c, 0) for c in classes}
    svh_dist = {c: svh_counts.get(c, 0) for c in classes}

    cross_n = pd.DataFrame(0, index=classes, columns=classes)
    for (g, s), cnt in cross_counts.items():
        cross_n.loc[g, s] = cnt
    cross_n["Total"] = cross_n.sum(axis=1)
    cross_n.loc["Total"] = cross_n.sum(axis=0)
    cross_pct = cross_n / N * 100

    both_rec = cross_counts.get(("Recovery", "Recovery"), 0)
    gpp_rec_svh_stab = cross_counts.get(("Recovery", "Stable"), 0)
    svh_rec_gpp_stab = cross_counts.get(("Stable", "Recovery"), 0)
    any_rec = sum(v for (g, s), v in cross_counts.items() if g == "Recovery" or s == "Recovery")
    both_deg = cross_counts.get(("Degradation", "Degradation"), 0)
    both_stab = cross_counts.get(("Stable", "Stable"), 0)

    all_ecos = sorted(eco_totals_nat.keys())
    eco_gpp = pd.DataFrame(0.0, index=all_ecos, columns=classes)
    eco_svh = pd.DataFrame(0.0, index=all_ecos, columns=classes)
    eco_comb = pd.DataFrame(0.0, index=all_ecos, columns=["n", "both_rec", "any_rec", "both_deg"])
    for eco in all_ecos:
        et = eco_totals_nat[eco]
        eco_comb.loc[eco, "n"] = et
        for c in classes:
            eco_gpp.loc[eco, c] = eco_gpp_counts.get((eco, c), 0) / et * 100
            eco_svh.loc[eco, c] = eco_svh_counts.get((eco, c), 0) / et * 100
        eco_comb.loc[eco, "both_rec"] = eco_cross_counts.get((eco, "Recovery", "Recovery"), 0) / et * 100
        eco_comb.loc[eco, "both_deg"] = eco_cross_counts.get((eco, "Degradation", "Degradation"), 0) / et * 100
        ar = sum(v for (e, g, s), v in eco_cross_counts.items() if e == eco and (g == "Recovery" or s == "Recovery"))
        eco_comb.loc[eco, "any_rec"] = ar / et * 100

    # ── 5. Write report ──
    print("Writing report...")
    L = []
    L.append("# Trajectory Recovery Analysis Report\n")
    L.append(f"**Date**: 2026-03-10  ")
    L.append(f"**Dataset**: `abandoned_ag_gpp_2000_2022_SA.parquet`  ")
    L.append(f"**Time series**: GPP and SVH, {YEARS[0]}-{YEARS[-1]} ({len(YEARS)} years)  ")
    L.append(f"**Method**: Mann-Kendall trend test + Theil-Sen slope (per-ecoregion Z-score standardized)  ")
    L.append(f"**Significance level**: alpha = 0.05\n")

    L.append("## 1. Data Overview\n")
    L.append(f"- **Total pixels**: {total_n:,}")
    L.append(f"- **Natural-LC pixels (SANLC 1-2)**: {filtered_n:,} ({filtered_n/total_n*100:.1f}%)")
    L.append(f"- **Transformed-LC pixels dropped**: {total_n - filtered_n:,} ({(total_n-filtered_n)/total_n*100:.1f}%)\n")

    L.append("### Overall land cover distribution\n")
    L.append("| SANLC Code | Land Cover | n | % |")
    L.append("|---:|---|---:|---:|")
    for _, row in lc_dist_all.dropna(subset=["sanlc_2022"]).iterrows():
        code = int(row["sanlc_2022"])
        label = SANLC_LABELS.get(code, "Unknown")
        nn = int(row["n"])
        L.append(f"| {code} | {label} | {nn:,} | {nn/total_n*100:.1f}% |")
    L.append("")

    L.append("### Land cover composition by ecoregion (%)\n")
    cols = sorted([c for c in lc_eco_pivot.columns if c != "Total pixels"]) + ["Total pixels"]
    L.append("| Ecoregion | " + " | ".join(cols) + " |")
    L.append("|---|" + "---:|" * len(cols))
    for eco in sorted(lc_eco_pivot.index):
        row = lc_eco_pivot.loc[eco]
        vals = [f"{int(row[c]):,}" if c == "Total pixels" else f"{row.get(c, 0):.1f}%" for c in cols]
        L.append(f"| {ename(eco)} | " + " | ".join(vals) + " |")
    L.append("")

    L.append("## 2. Functional Trajectory (GPP)\n")
    L.append("Classification based on Mann-Kendall significance (p < 0.05) and Theil-Sen slope direction.\n")
    L.append("| Class | n | % |")
    L.append("|---|---:|---:|")
    for c in classes:
        L.append(f"| {c} | {gpp_dist[c]:,} | {gpp_dist[c]/N*100:.1f}% |")
    L.append("")

    L.append("## 3. Structural Trajectory (SVH)\n")
    L.append("| Class | n | % |")
    L.append("|---|---:|---:|")
    for c in classes:
        L.append(f"| {c} | {svh_dist[c]:,} | {svh_dist[c]/N*100:.1f}% |")
    L.append("")

    L.append("## 4. Combined Functional x Structural Classification\n")
    L.append("Cross-tabulation of GPP and SVH trajectory classes.\n")

    L.append("### Counts\n")
    order = ["Recovery", "Stable", "Degradation", "Total"]
    L.append("| GPP \\ SVH | " + " | ".join(order) + " |")
    L.append("|---|" + "---:|" * len(order))
    for g in order:
        vals = [f"{int(cross_n.loc[g, s]):,}" for s in order]
        L.append(f"| **{g}** | " + " | ".join(vals) + " |")
    L.append("")

    L.append("### Percentages\n")
    L.append("| GPP \\ SVH | " + " | ".join(order) + " |")
    L.append("|---|" + "---:|" * len(order))
    for g in order:
        vals = [f"{cross_pct.loc[g, s]:.2f}%" for s in order]
        L.append(f"| **{g}** | " + " | ".join(vals) + " |")
    L.append("")

    L.append("### Combined class distribution\n")
    L.append("| Combined Class | n | % |")
    L.append("|---|---:|---:|")
    combos = sorted(cross_counts.keys(), key=lambda x: (x[0], x[1]))
    for (g, s) in combos:
        cnt = cross_counts[(g, s)]
        L.append(f"| {g} + {s} | {cnt:,} | {cnt/N*100:.1f}% |")
    L.append("")

    L.append("### Key recovery metrics\n")
    L.append(f"- **Both recovering** (GPP + SVH): {both_rec:,} ({both_rec/N*100:.1f}%)")
    L.append(f"- **Functional recovery only** (GPP recovering, SVH stable): {gpp_rec_svh_stab:,} ({gpp_rec_svh_stab/N*100:.1f}%)")
    L.append(f"- **Structural recovery only** (SVH recovering, GPP stable): {svh_rec_gpp_stab:,} ({svh_rec_gpp_stab/N*100:.1f}%)")
    L.append(f"- **Any recovery** (at least one recovering): {any_rec:,} ({any_rec/N*100:.1f}%)")
    L.append(f"- **Both stable**: {both_stab:,} ({both_stab/N*100:.1f}%)")
    L.append(f"- **Both degrading**: {both_deg:,} ({both_deg/N*100:.1f}%)\n")

    L.append("## 5. Ecoregion Breakdown\n")
    L.append("### GPP trajectory by ecoregion (%)\n")
    L.append("| Ecoregion | n | Recovery | Stable | Degradation |")
    L.append("|---|---:|---:|---:|---:|")
    for eco in all_ecos:
        ne = int(eco_comb.loc[eco, "n"])
        L.append(f"| {ename(eco)} | {ne:,} | {eco_gpp.loc[eco, 'Recovery']:.1f}% | {eco_gpp.loc[eco, 'Stable']:.1f}% | {eco_gpp.loc[eco, 'Degradation']:.1f}% |")
    L.append("")

    L.append("### SVH trajectory by ecoregion (%)\n")
    L.append("| Ecoregion | n | Recovery | Stable | Degradation |")
    L.append("|---|---:|---:|---:|---:|")
    for eco in all_ecos:
        ne = int(eco_comb.loc[eco, "n"])
        L.append(f"| {ename(eco)} | {ne:,} | {eco_svh.loc[eco, 'Recovery']:.1f}% | {eco_svh.loc[eco, 'Stable']:.1f}% | {eco_svh.loc[eco, 'Degradation']:.1f}% |")
    L.append("")

    L.append("### Combined recovery by ecoregion\n")
    L.append("| Ecoregion | n | Both recovering (%) | Any recovering (%) | Both degrading (%) |")
    L.append("|---|---:|---:|---:|---:|")
    for eco in all_ecos:
        r = eco_comb.loc[eco]
        L.append(f"| {ename(eco)} | {int(r['n']):,} | {r['both_rec']:.1f}% | {r['any_rec']:.1f}% | {r['both_deg']:.1f}% |")
    L.append("")

    L.append("## 6. Slope Statistics\n")
    L.append("Mean Theil-Sen slopes by GPP trajectory class.\n")
    L.append("| GPP Class | GPP slope (mean) | SVH slope (mean) |")
    L.append("|---|---:|---:|")
    for c in classes:
        gm = slope_sums[c]["gpp"] / max(slope_sums[c]["n"], 1)
        sm = slope_sums[c]["svh"] / max(slope_sums[c]["n"], 1)
        L.append(f"| {c} | {gm:.5f} | {sm:.5f} |")
    L.append("")

    L.append("---\n")
    L.append("*Analysis limited to pixels with natural/near-natural or secondary natural land cover (SANLC 2022 classes 1-2). "
             "Trends computed on per-ecoregion Z-score standardized values. "
             "Classification: significant positive slope = Recovery, significant negative slope = Degradation, "
             "non-significant = Stable (alpha = 0.05).*\n")

    report_text = "\n".join(L)
    REPORT.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved: {REPORT}")
    print("\n" + "=" * 60)
    print(report_text)


if __name__ == "__main__":
    main()
