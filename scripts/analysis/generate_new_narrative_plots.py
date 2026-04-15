"""
Generate additional narrative plots for the presentation:
1. Metric decomposition radar chart for example pixels A-D
2. Recovery score violin distributions by ecoregion
3. Filtering funnel / waterfall chart

Usage:
    python scripts/analysis/generate_new_narrative_plots.py
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import duckdb
import os
import warnings

warnings.filterwarnings("ignore")

out_dir = r"c:\Users\coach\myfiles\postdoc2\code\plots\narrative"
data_dir = r"c:\Users\coach\myfiles\postdoc2\code\data"
os.makedirs(out_dir, exist_ok=True)

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


# ============================================================
# 1. RADAR CHART — Metric decomposition for example pixels
# ============================================================
def plot_metric_radar():
    """Radar/spider chart showing all 6 metric percentiles for pixels A-D."""

    # Data from recovery_scores queries (exact lat/lon matched rows)
    pixels = {
        "C — Early\n(24.1)": {
            "Ecoregion\nGPP": 50.8, "Ecoregion\nSVH": 41.9,
            "Local\nGPP": 55.8, "Local\nSVH": 1.9,
            "Embedding\nEcoregion": 20.2, "Embedding\nLocal": 1.0,
        },
        "A — Moderate\n(53.6)": {
            "Ecoregion\nGPP": 91.0, "Ecoregion\nSVH": 79.7,
            "Local\nGPP": 84.9, "Local\nSVH": 79.3,
            "Embedding\nEcoregion": 30.8, "Embedding\nLocal": 16.3,
        },
        "B — Advanced\n(82.2)": {
            "Ecoregion\nGPP": 98.0, "Ecoregion\nSVH": 99.2,
            "Local\nGPP": 90.5, "Local\nSVH": 99.5,
            "Embedding\nEcoregion": 83.8, "Embedding\nLocal": 51.3,
        },
        "D — Full\n(94.1)": {
            "Ecoregion\nGPP": 93.9, "Ecoregion\nSVH": 94.1,
            "Local\nGPP": 95.4, "Local\nSVH": 96.2,
            "Embedding\nEcoregion": 98.8, "Embedding\nLocal": 87.7,
        },
    }

    categories = list(next(iter(pixels.values())).keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    colors = ["#95a5a6", "#e67e22", "#3498db", "#27ae60"]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for (label, metrics), color in zip(pixels.items(), colors):
        values = list(metrics.values())
        values += values[:1]  # close
        ax.plot(angles, values, "o-", linewidth=2.5, label=label, color=color,
                markersize=7)
        ax.fill(angles, values, alpha=0.08, color=color)

    # Natural median reference ring
    median_ring = [50] * (N + 1)
    ax.plot(angles, median_ring, "--", color="#e74c3c", linewidth=2, alpha=0.7,
            label="Natural Median (50)")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=16, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=14, color="gray")
    ax.yaxis.set_label_position("left")

    # Legend outside
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=15,
              frameon=True, shadow=True, title="Pixel (Score)",
              title_fontsize=16)

    plt.title("Recovery Metric Decomposition\nExample Pixels A–D", pad=30,
              fontsize=24, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metric_radar_examples.png"), dpi=300,
                bbox_inches="tight")
    plt.close()
    print("[OK] metric_radar_examples.png")


# ============================================================
# 2. VIOLIN PLOT — Recovery score distributions by ecoregion
# ============================================================
def plot_score_distributions():
    """Violin plot of recovery_score distributions for top ecoregions."""

    eco_ids = [81, 41, 38, 40, 48, 89, 90, 88, 16, 97, 101, 110, 102, 19, 94, 15, 116, 65]
    eco_names = {
        81: "Highveld Grasslands",
        41: "Drakensberg Grasslands",
        38: "Central Bushveld",
        40: "KwaZulu-Cape Coastal",
        48: "Maputaland Coastal",
        89: "Fynbos",
        90: "Succulent Karoo",
        88: "Albany Thickets",
        16: "E. Zimbabwe Montane",
        97: "Kalahari Xeric Savanna",
        101: "Knysna-Amatole Forests",
        110: "Zambezian Woodlands",
        102: "Namaqualand-Richtersveld",
        19: "Maputaland-Pondoland",
        94: "S. Africa Bushveld",
        15: "Lowland Fynbos",
        116: "Montane Fynbos",
        65: "Nama Karoo",
    }

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")

    frames = []
    for eid in eco_ids:
        path = os.path.join(data_dir, f"recovery_scores_eco{eid}.parquet")
        if not os.path.exists(path):
            continue
        # Sample up to 5000 per ecoregion to keep the violin manageable
        df = con.sql(f"""
            SELECT recovery_score, {eid} as eco_id
            FROM '{path}'
            WHERE recovery_score IS NOT NULL
            USING SAMPLE reservoir(5000 ROWS) REPEATABLE(42)
        """).df()
        df["ecoregion"] = eco_names.get(eid, str(eid))
        frames.append(df)

    con.close()
    all_df = pd.concat(frames, ignore_index=True)

    # Order by median score
    order = (all_df.groupby("ecoregion")["recovery_score"]
             .median().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.violinplot(
        data=all_df, y="ecoregion", x="recovery_score", order=order,
        palette="viridis_r", inner="quartile", linewidth=1.2, cut=0,
        density_norm="width", ax=ax,
    )

    # Natural median reference line
    ax.axvline(50, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.8,
               label="Natural Median (50)")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Composite Recovery Score (0–100)", labelpad=15)
    ax.set_ylabel("")
    ax.set_title("Recovery Score Distribution by Ecoregion", pad=20,
                 fontweight="bold")
    ax.legend(loc="lower right", frameon=True, shadow=True)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recovery_score_distributions.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] recovery_score_distributions.png")


# ============================================================
# 3. FUNNEL / WATERFALL — Progressive filtering cascade
# ============================================================
def plot_recovery_funnel():
    """Waterfall chart showing progressive data narrowing from raw to native recovery."""

    stages = [
        ("Raw Abandoned\nPixels", 38_500_000, None),
        ("SANLC/Morphology\nFiltering", 33_400_000, "Removed transformed LC\n(−5.1M, −13.2%)"),
        ("Dual-Trend\nRecovery Filter", 2_881_285, "No significant trend\nin GPP & SVH\n(−30.5M, −91.4%)"),
        ("NIAPS Invasive\nExclusion", 2_220_068, "Alien-invaded pixels\n(−661K, −22.9%)"),
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Bar positions
    x = np.arange(len(stages))
    bar_width = 0.55
    colors = ["#bdc3c7", "#3498db", "#e67e22", "#27ae60"]

    bars = ax.bar(x, [s[1] for s in stages], width=bar_width, color=colors,
                  edgecolor="black", linewidth=1.2, zorder=3)

    # Value labels on bars
    for i, (label, count, _) in enumerate(stages):
        if count >= 1_000_000:
            txt = f"{count/1_000_000:.1f}M"
        else:
            txt = f"{count/1_000:,.0f}K"

        # Place label inside or above bar
        bar_h = bars[i].get_height()
        if bar_h > stages[0][1] * 0.15:
            ax.text(i, bar_h / 2, txt, ha="center", va="center",
                    fontsize=22, fontweight="bold", color="white", zorder=4)
        else:
            ax.text(i, bar_h + stages[0][1] * 0.015, txt, ha="center",
                    va="bottom", fontsize=20, fontweight="bold", color="black",
                    zorder=4)

    # Arrows between bars showing what was removed
    # Manual y-positions for annotations to avoid overlap
    annotation_y = [None, 36_000_000, 28_000_000, 4_500_000]
    for i in range(1, len(stages)):
        if stages[i][2]:
            mid_x = (x[i - 1] + x[i]) / 2
            # Draw arrow from previous bar top to current bar top
            ax.annotate(
                "", xy=(x[i], stages[i][1] + stages[0][1] * 0.01),
                xytext=(x[i - 1], stages[i - 1][1]),
                arrowprops=dict(arrowstyle="-|>", color="gray", lw=2,
                                connectionstyle="arc3,rad=-0.15"),
                zorder=2,
            )
            # Removal annotation — placed at manually tuned y positions
            ax.text(mid_x, annotation_y[i],
                    stages[i][2], ha="center", va="center", fontsize=12,
                    color="#555555", style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.9, edgecolor="#cccccc"))

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in stages], fontsize=16)
    ax.set_ylabel("Number of 30m Pixels", labelpad=15)
    ax.set_title("Progressive Filtering: From Abandoned Land to Native Recovery",
                 pad=20, fontweight="bold")
    ax.set_ylim(0, stages[0][1] * 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda val, pos: f"{val/1e6:.0f}M" if val >= 1e6 else f"{val/1e3:.0f}K"))
    sns.despine(left=False, bottom=True)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recovery_funnel.png"), dpi=300,
                bbox_inches="tight")
    plt.close()
    print("[OK] recovery_funnel.png")


# ============================================================
# 4. SCATTER — Metric A vs C and Metric B vs C (two-panel)
# ============================================================
def plot_metric_scatter_panels():
    """Two-panel scatter: (left) A vs C, (right) B vs C.

    Left panel: ecoregion-level function/structure vs embedding similarity.
    Right panel: local-level function/structure vs embedding similarity.
    Demonstrates that both ecoregion and local productivity are decoupled
    from latent ecological similarity.
    """

    eco_ids = [81, 41, 38, 40, 48, 89, 90, 88, 16, 97, 101, 110, 102, 19, 94, 15, 116, 65]
    eco_names = {
        81: "Highveld Grasslands", 41: "Drakensberg Grasslands",
        38: "Central Bushveld", 40: "KwaZulu-Cape Coastal",
        48: "Maputaland Coastal", 89: "Fynbos",
        90: "Succulent Karoo", 88: "Albany Thickets",
        16: "E. Zimbabwe Montane", 97: "Kalahari Xeric Savanna",
        101: "Knysna-Amatole Forests", 110: "Zambezian Woodlands",
        102: "Namaqualand-Richtersveld", 19: "Maputaland-Pondoland",
        94: "S. Africa Bushveld", 15: "Lowland Fynbos",
        116: "Montane Fynbos", 65: "Nama Karoo",
    }

    highlight_ecos = [81, 40, 41, 38, 48, 89]

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")

    frames = []
    for eid in eco_ids:
        path = os.path.join(data_dir, f"recovery_scores_eco{eid}.parquet")
        if not os.path.exists(path):
            continue
        n_sample = 3000 if eid in highlight_ecos else 500
        df = con.sql(f"""
            SELECT
                (a_gpp_pctl + a_svh_pctl) / 2.0 AS metric_a,
                (b_gpp_pctl + b_svh_pctl) / 2.0 AS metric_b,
                (c_eco_pctl + c_local_pctl) / 2.0 AS metric_c,
                {eid} AS eco_id
            FROM '{path}'
            WHERE a_gpp_pctl IS NOT NULL
              AND c_eco_pctl IS NOT NULL
              AND b_gpp_pctl IS NOT NULL
            USING SAMPLE reservoir({n_sample} ROWS) REPEATABLE(42)
        """).df()
        if eid in highlight_ecos:
            df["ecoregion"] = eco_names[eid]
        else:
            df["ecoregion"] = "Other"
        frames.append(df)

    con.close()
    all_df = pd.concat(frames, ignore_index=True)

    palette = {
        "Highveld Grasslands": "#2ecc71",
        "KwaZulu-Cape Coastal": "#e74c3c",
        "Drakensberg Grasslands": "#3498db",
        "Central Bushveld": "#e67e22",
        "Maputaland Coastal": "#9b59b6",
        "Fynbos": "#1abc9c",
        "Other": "#bdc3c7",
    }
    order = ["Other"] + [eco_names[e] for e in highlight_ecos]

    # Example pixel metric values
    # Pixel A: A=(91+79.7)/2=85.3, B=(84.9+79.3)/2=82.1, C=(30.8+16.3)/2=23.6
    # Pixel D: A=(93.9+94.1)/2=94.0, B=(95.4+96.2)/2=95.8, C=(98.8+87.7)/2=93.3
    pixel_examples = {
        "A": {"a": 85.3, "b": 82.1, "c": 23.6, "color": "#e67e22"},
        "D": {"a": 94.0, "b": 95.8, "c": 93.3, "color": "#27ae60"},
    }

    # --- Build panels ---
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharex=False, sharey=True)

    panel_configs = [
        ("metric_a", "Metric A: Ecoregion GPP + SVH", "a"),
        ("metric_b", "Metric B: Local GPP + SVH", "b"),
    ]

    for ax, (x_col, xlabel_short, px_key) in zip(axes, panel_configs):
        for eco_label in order:
            subset = all_df[all_df["ecoregion"] == eco_label]
            alpha = 0.15 if eco_label == "Other" else 0.35
            size = 8 if eco_label == "Other" else 14
            ax.scatter(subset[x_col], subset["metric_c"],
                       c=palette[eco_label], alpha=alpha, s=size,
                       edgecolors="none", label=eco_label,
                       zorder=2 if eco_label == "Other" else 3)

        # Reference lines
        ax.axhline(50, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.5, zorder=1)
        ax.axvline(50, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.5, zorder=1)
        ax.plot([0, 100], [0, 100], "-", color="gray", linewidth=1, alpha=0.4, zorder=1)

        # Quadrant labels
        ax.text(85, 8, '"Green but\n Different"', fontsize=15, ha="center",
                va="center", color="#7f8c8d", style="italic", alpha=0.7)
        ax.text(85, 92, "Full\nRecovery", fontsize=15, ha="center",
                va="center", color="#27ae60", style="italic", alpha=0.7)
        ax.text(15, 8, "Early\nRecovery", fontsize=15, ha="center",
                va="center", color="#95a5a6", style="italic", alpha=0.7)

        # Example pixels
        for plabel, pdata in pixel_examples.items():
            px_x = pdata[px_key]
            px_y = pdata["c"]
            ax.scatter([px_x], [px_y], s=250, c=pdata["color"],
                       edgecolors="black", linewidth=2, zorder=5, marker="*")
            ax.annotate(f"Pixel {plabel}", xy=(px_x, px_y),
                        xytext=(px_x - 10, px_y + 7),
                        fontsize=13, fontweight="bold", color=pdata["color"],
                        arrowprops=dict(arrowstyle="-", color=pdata["color"], lw=1.5))

        # Correlation
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

    axes[0].set_ylabel("Mean Metric C Percentile\n(Embedding Similarity to Natural)",
                       labelpad=12, fontsize=18)

    # Shared legend from the left panel
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)
    fig.legend(unique_handles, unique_labels,
               loc="lower center", ncol=7, fontsize=12,
               frameon=True, shadow=True, markerscale=3,
               title="Ecoregion", title_fontsize=13,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Functional/Structural Recovery vs Ecological Identity",
                 fontsize=24, fontweight="bold", y=1.01)
    plt.subplots_adjust(bottom=0.15, wspace=0.08)
    plt.savefig(os.path.join(out_dir, "metric_a_vs_c_scatter.png"), dpi=300,
                bbox_inches="tight")
    plt.close()
    print("[OK] metric_a_vs_c_scatter.png (2-panel: A vs C | B vs C)")


if __name__ == "__main__":
    plot_metric_radar()
    plot_score_distributions()
    plot_recovery_funnel()
    plot_metric_scatter_panels()
    print("\nAll new narrative plots generated.")
