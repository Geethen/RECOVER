"""
Generate narrative plots for recovery assessment stratified by bioregion.

Adapted from generate_narrative_plots.py and generate_new_narrative_plots.py.
Reads recovery_scores_bio{id}.parquet files and produces publication-quality
charts grouped by bioregion.

Prerequisite: run score_by_bioregion.py first.

Usage:
    python scripts/analysis/generate_bioregion_narrative_plots.py
"""
import json
import os
import warnings

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

warnings.filterwarnings("ignore")

BASE_DIR = r"c:\Users\coach\myfiles\postdoc2\code"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "plots", "narrative_bioregion")
os.makedirs(OUT_DIR, exist_ok=True)

# Publication style
sns.set_theme(style="ticks", context="talk", font="sans-serif")
plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 26
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["axes.linewidth"] = 1.5


def load_bioregion_names():
    """Load bioregion ID → name mapping from metadata JSON."""
    meta_path = r"c:\Users\coach\myfiles\postdoc2\data\bioregions_30m.json"
    with open(meta_path) as f:
        meta = json.load(f)
    names = {}
    for entry in meta["bioregions"]:
        bid = entry["id"]
        name = entry["name"]
        if bid not in names or name != "None":
            names[bid] = name
    return names


def discover_bio_files():
    """Find all recovery_scores_bio{id}.parquet files and return list of bio_ids."""
    import glob
    pattern = os.path.join(DATA_DIR, "recovery_scores_bio*.parquet")
    files = glob.glob(pattern)
    bio_ids = []
    for f in files:
        base = os.path.basename(f)
        # Extract ID from recovery_scores_bio{id}.parquet
        bid = int(base.replace("recovery_scores_bio", "").replace(".parquet", ""))
        bio_ids.append(bid)
    return sorted(bio_ids)


def load_summary_data(bio_ids, bio_names):
    """Load summary statistics per bioregion from parquet files."""
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")

    rows = []
    for bid in bio_ids:
        path = os.path.join(DATA_DIR, f"recovery_scores_bio{bid}.parquet")
        if not os.path.exists(path):
            continue
        stats = con.sql(f"""
            SELECT
                count(*) as total,
                median(recovery_score) as median_score,
                sum(CASE WHEN niaps = 1 THEN 1 ELSE 0 END) as n_invasive
            FROM '{path}'
        """).fetchone()

        total, median_score, n_invasive = stats
        if total == 0:
            continue
        n_invasive = n_invasive or 0
        pct_inv = 100 * n_invasive / total

        name = bio_names.get(bid, f"Bioregion {bid}")
        rows.append({
            "Bioregion": name,
            "ID": bid,
            "TotalPixels": total,
            "RecoveryScore": float(median_score),
            "InvasivePixels": n_invasive,
            "PctInvasive": pct_inv,
        })

    con.close()
    return pd.DataFrame(rows)


# ============================================================
# 1. Recovery Score by Bioregion (horizontal bar)
# ============================================================
def plot_recovery_scores(df):
    fig_height = max(8, len(df) * 0.5)
    plt.figure(figsize=(14, fig_height))
    df_sorted = df.sort_values(by="RecoveryScore", ascending=True)
    cmap = sns.color_palette("viridis_r", len(df_sorted))
    bars = sns.barplot(data=df_sorted, x="RecoveryScore", y="Bioregion",
                       hue="Bioregion", palette=cmap, legend=False)
    plt.xlim(0, df_sorted["RecoveryScore"].max() * 1.15)
    plt.axvline(50, color="#e74c3c", linestyle="--", linewidth=2,
                label="Natural Median (50)")
    sns.despine(left=True, bottom=False)

    for p in bars.patches:
        width = p.get_width()
        if width > 0:
            plt.text(width + 0.5, p.get_y() + p.get_height() / 2,
                     f"{width:.1f}", ha="left", va="center", fontsize=16,
                     color="black", alpha=0.8)

    plt.title("Median Ecological Recovery Score by Bioregion",
              pad=20, fontweight="bold")
    plt.xlabel("Composite Recovery Score (0–100)", labelpad=15)
    plt.ylabel("")
    plt.legend(loc="lower left", frameon=True, shadow=True,
               bbox_to_anchor=(0.02, 0.02))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "recovery_scores_by_bioregion.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] recovery_scores_by_bioregion.png")


# ============================================================
# 2. Invasive Plants by Bioregion (horizontal bar)
# ============================================================
def plot_invasive_plants(df):
    fig_height = max(8, len(df) * 0.5)
    plt.figure(figsize=(14, fig_height))
    df_sorted = df.sort_values(by="PctInvasive", ascending=True)
    cmap = sns.color_palette("Reds", len(df_sorted))
    bars = sns.barplot(data=df_sorted, x="PctInvasive", y="Bioregion",
                       hue="Bioregion", palette=cmap, legend=False)
    plt.xlim(0, max(df_sorted["PctInvasive"].max() * 1.15, 5))
    sns.despine(left=True, bottom=False)

    for p in bars.patches:
        width = p.get_width()
        if width > 0:
            plt.text(width + 0.3, p.get_y() + p.get_height() / 2,
                     f"{width:.1f}%", ha="left", va="center", fontsize=16,
                     color="black", alpha=0.8)

    plt.title("Recovering Pixels Invaded by Alien Plants",
              pad=20, fontweight="bold")
    plt.xlabel("Percentage Invaded (%)", labelpad=15)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "invasive_plants_by_bioregion.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] invasive_plants_by_bioregion.png")


# ============================================================
# 3. Recovery Score vs Invasive Scatter (bubble)
# ============================================================
def plot_recovery_vs_invasive(df):
    plt.figure(figsize=(16, 10))
    scatter = sns.scatterplot(
        data=df, x="PctInvasive", y="RecoveryScore", size="TotalPixels",
        sizes=(100, 2500), alpha=0.6, color="#2ecc71", edgecolor="black",
        linewidth=1)

    texts = []
    for i in range(df.shape[0]):
        name = df.iloc[i]["Bioregion"]
        show = (df.iloc[i]["TotalPixels"] > df["TotalPixels"].quantile(0.7)
                or df.iloc[i]["PctInvasive"] > df["PctInvasive"].quantile(0.8)
                or df.iloc[i]["RecoveryScore"] < df["RecoveryScore"].quantile(0.15)
                or df.iloc[i]["RecoveryScore"] > df["RecoveryScore"].quantile(0.85))
        if show:
            if len(name) > 25:
                name = name[:22] + "..."
            t = plt.text(
                df.iloc[i]["PctInvasive"] + 0.3,
                df.iloc[i]["RecoveryScore"] + 0.3,
                name, fontsize=14, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3))
            texts.append(t)

    if texts:
        adjust_text(texts,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=1.2),
                    expand_points=(2.0, 2.0),
                    force_points=(2.5, 2.5),
                    force_text=(1.5, 1.5))

    plt.axhline(50, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.8,
                label="Natural Median (50)")
    sns.despine(bottom=False, left=False)
    plt.title("Recovery Score vs. Alien Plant Invasion by Bioregion",
              pad=20, fontweight="bold")
    plt.xlabel("Percentage Invaded by Alien Plants (%)", labelpad=15)
    plt.ylabel("Median Composite Recovery Score (0–100)", labelpad=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.,
               title="Recovering Pixels", shadow=True, markerscale=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "recovery_vs_invasive.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] recovery_vs_invasive.png")


# ============================================================
# 4. Top Bioregions by Recovering Pixel Count (bar)
# ============================================================
def plot_pixels_by_bioregion(df, top_n=15):
    plt.figure(figsize=(14, max(8, top_n * 0.5)))
    df_top = df.sort_values(by="TotalPixels", ascending=False).head(top_n)
    df_top = df_top.sort_values(by="TotalPixels", ascending=True)
    cmap = sns.color_palette("Greens_r", len(df_top))
    bars = sns.barplot(data=df_top, x="TotalPixels", y="Bioregion",
                       hue="Bioregion", palette=cmap, legend=False)
    plt.xlim(0, df_top["TotalPixels"].max() * 1.15)
    sns.despine(left=True, bottom=False)

    max_width = df_top["TotalPixels"].max()
    for p in bars.patches:
        width = p.get_width()
        if width > 0:
            if width > max_width * 0.15:
                plt.text(width - (max_width * 0.02),
                         p.get_y() + p.get_height() / 2,
                         f"{width / 1000:,.0f}k", ha="right", va="center",
                         fontsize=16, color="white", fontweight="bold")
            else:
                plt.text(width + (max_width * 0.01),
                         p.get_y() + p.get_height() / 2,
                         f"{width / 1000:,.0f}k", ha="left", va="center",
                         fontsize=16, color="black", alpha=0.8)

    plt.title(f"Top {top_n} Bioregions by Recovering Pixel Count",
              pad=20, fontweight="bold")
    plt.xlabel("Number of Recovering Pixels", labelpad=15)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pixels_by_bioregion.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] pixels_by_bioregion.png")


# ============================================================
# 5. Violin — Recovery score distributions by bioregion
# ============================================================
def plot_score_distributions(bio_ids, bio_names):
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")

    frames = []
    for bid in bio_ids:
        path = os.path.join(DATA_DIR, f"recovery_scores_bio{bid}.parquet")
        if not os.path.exists(path):
            continue
        df = con.sql(f"""
            SELECT recovery_score, {bid} as bio_id
            FROM '{path}'
            WHERE recovery_score IS NOT NULL
            USING SAMPLE reservoir(5000 ROWS) REPEATABLE(42)
        """).df()
        name = bio_names.get(bid, f"Bioregion {bid}")
        df["bioregion"] = name
        frames.append(df)

    con.close()
    if not frames:
        print("[SKIP] No data for violin plot")
        return

    all_df = pd.concat(frames, ignore_index=True)
    order = (all_df.groupby("bioregion")["recovery_score"]
             .median().sort_values().index.tolist())

    fig_height = max(10, len(order) * 0.6)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    sns.violinplot(
        data=all_df, y="bioregion", x="recovery_score", order=order,
        palette="viridis_r", inner="quartile", linewidth=1.2, cut=0,
        density_norm="width", ax=ax)

    ax.axvline(50, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.8,
               label="Natural Median (50)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Composite Recovery Score (0–100)", labelpad=15)
    ax.set_ylabel("")
    ax.set_title("Recovery Score Distribution by Bioregion",
                 pad=20, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, shadow=True)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "recovery_score_distributions.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] recovery_score_distributions.png")


# ============================================================
# 6. Scatter — Metric A vs C and B vs C (two-panel)
# ============================================================
def plot_metric_scatter_panels(bio_ids, bio_names):
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")

    # Pick top 6 bioregions by pixel count for highlighting
    counts = {}
    for bid in bio_ids:
        path = os.path.join(DATA_DIR, f"recovery_scores_bio{bid}.parquet")
        if not os.path.exists(path):
            continue
        n = con.sql(f"SELECT count(*) FROM '{path}'").fetchone()[0]
        counts[bid] = n

    top6 = sorted(counts, key=counts.get, reverse=True)[:6]

    frames = []
    for bid in bio_ids:
        path = os.path.join(DATA_DIR, f"recovery_scores_bio{bid}.parquet")
        if not os.path.exists(path):
            continue
        n_sample = 3000 if bid in top6 else 500
        df = con.sql(f"""
            SELECT
                (a_gpp_pctl + a_svh_pctl) / 2.0 AS metric_a,
                (b_gpp_pctl + b_svh_pctl) / 2.0 AS metric_b,
                (c_eco_pctl + c_local_pctl) / 2.0 AS metric_c,
                {bid} AS bio_id
            FROM '{path}'
            WHERE a_gpp_pctl IS NOT NULL
              AND c_eco_pctl IS NOT NULL
              AND b_gpp_pctl IS NOT NULL
            USING SAMPLE reservoir({n_sample} ROWS) REPEATABLE(42)
        """).df()
        name = bio_names.get(bid, f"Bio {bid}")
        df["bioregion"] = name if bid in top6 else "Other"
        frames.append(df)

    con.close()
    if not frames:
        print("[SKIP] No data for scatter plot")
        return

    all_df = pd.concat(frames, ignore_index=True)

    # Build palette
    highlight_names = [bio_names.get(b, f"Bio {b}") for b in top6]
    base_colors = ["#2ecc71", "#e74c3c", "#3498db", "#e67e22", "#9b59b6", "#1abc9c"]
    palette = {"Other": "#bdc3c7"}
    for name, color in zip(highlight_names, base_colors):
        palette[name] = color
    order = ["Other"] + highlight_names

    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharex=False, sharey=True)

    panel_configs = [
        ("metric_a", "Metric A: Bioregion GPP + SVH"),
        ("metric_b", "Metric B: Local GPP + SVH"),
    ]

    for ax, (x_col, xlabel_short) in zip(axes, panel_configs):
        for label in order:
            subset = all_df[all_df["bioregion"] == label]
            alpha = 0.15 if label == "Other" else 0.35
            size = 8 if label == "Other" else 14
            ax.scatter(subset[x_col], subset["metric_c"],
                       c=palette.get(label, "#bdc3c7"), alpha=alpha, s=size,
                       edgecolors="none", label=label,
                       zorder=2 if label == "Other" else 3)

        ax.axhline(50, color="#e74c3c", linestyle="--", linewidth=1.5,
                   alpha=0.5, zorder=1)
        ax.axvline(50, color="#e74c3c", linestyle="--", linewidth=1.5,
                   alpha=0.5, zorder=1)
        ax.plot([0, 100], [0, 100], "-", color="gray", linewidth=1,
                alpha=0.4, zorder=1)

        ax.text(85, 8, '"Green but\n Different"', fontsize=15, ha="center",
                va="center", color="#7f8c8d", style="italic", alpha=0.7)
        ax.text(85, 92, "Full\nRecovery", fontsize=15, ha="center",
                va="center", color="#27ae60", style="italic", alpha=0.7)
        ax.text(15, 8, "Early\nRecovery", fontsize=15, ha="center",
                va="center", color="#95a5a6", style="italic", alpha=0.7)

        corr_val = all_df[x_col].corr(all_df["metric_c"])
        ax.text(0.03, 0.97, f"r = {corr_val:.2f}", transform=ax.transAxes,
                fontsize=18, va="top", ha="left", color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.9))

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel(f"{xlabel_short} Percentile", labelpad=12, fontsize=16)
        ax.set_aspect("equal")
        ax.set_title(f"{xlabel_short} vs Embedding Similarity",
                     pad=15, fontsize=20, fontweight="bold")
        sns.despine(ax=ax)

    axes[0].set_ylabel(
        "Mean Metric C Percentile\n(Embedding Similarity to Natural)",
        labelpad=12, fontsize=18)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_h.append(h)
            unique_l.append(l)
    fig.legend(unique_h, unique_l,
               loc="lower center", ncol=min(7, len(unique_l)), fontsize=12,
               frameon=True, shadow=True, markerscale=3,
               title="Bioregion", title_fontsize=13,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Functional/Structural Recovery vs Ecological Identity",
                 fontsize=24, fontweight="bold", y=1.01)
    plt.subplots_adjust(bottom=0.15, wspace=0.08)
    plt.savefig(os.path.join(OUT_DIR, "metric_a_vs_c_scatter.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] metric_a_vs_c_scatter.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    bio_names = load_bioregion_names()
    bio_ids = discover_bio_files()

    if not bio_ids:
        print("[ERROR] No recovery_scores_bio*.parquet files found. "
              "Run score_by_bioregion.py first.")
        exit(1)

    print(f"Found {len(bio_ids)} bioregion score files: {bio_ids}")

    # Load summary data
    df = load_summary_data(bio_ids, bio_names)
    print(f"\nSummary table ({len(df)} bioregions):")
    print(df[["ID", "Bioregion", "TotalPixels", "RecoveryScore", "PctInvasive"]]
          .sort_values("TotalPixels", ascending=False).to_string(index=False))

    # Generate all plots
    print("\nGenerating plots ...")
    plot_recovery_scores(df)
    plot_invasive_plants(df)
    plot_recovery_vs_invasive(df)
    plot_pixels_by_bioregion(df)
    plot_score_distributions(bio_ids, bio_names)
    plot_metric_scatter_panels(bio_ids, bio_names)

    print(f"\nAll plots saved to: {OUT_DIR}")
