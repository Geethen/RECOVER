"""
Explore WPE as a context flag for recovery interpretation.

Key question: In which ecoregions and under what conditions does a significant
positive WPE trend meaningfully change the interpretation of recovery scores?

Focus on the interaction between:
  - WPE trend × sig_mask (significant woody increase)
  - SVH slope / SVH percentile (structural recovery signal)
  - Recovery score (composite)
  - Ecoregion type (grassland vs savanna vs forest)

Input:  data/wpe_niaps_calibration.parquet
Output: plots/wpe_context_*.png, printed summary
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
ECO_CSV = DATA_DIR / "ecoregion_sds.csv"

# Ecoregion groupings by vegetation type
GRASSLAND_ECOS = [81, 41]  # Highveld, Drakensberg
SAVANNA_ECOS = [38, 97, 94, 110]  # Central Bushveld, Kalahari, Gariep Karoo, Mopane
FOREST_ECOS = [40, 48, 16, 19, 101]  # coastal/montane forests
SHRUBLAND_ECOS = [89, 90, 88, 102, 15, 116]  # fynbos, karoo, thickets

BIOME_MAP = {}
for e in GRASSLAND_ECOS: BIOME_MAP[e] = "Grassland"
for e in SAVANNA_ECOS: BIOME_MAP[e] = "Savanna/Bushveld"
for e in FOREST_ECOS: BIOME_MAP[e] = "Forest mosaic"
for e in SHRUBLAND_ECOS: BIOME_MAP[e] = "Shrubland/Thicket"


def load_eco_names():
    df = pd.read_csv(ECO_CSV, usecols=["ECO_ID", "ECO_NAME"]).drop_duplicates("ECO_ID")
    return dict(zip(df["ECO_ID"].astype(int), df["ECO_NAME"]))


def main():
    eco_names = load_eco_names()
    df = pd.read_parquet(str(DATA_DIR / "wpe_niaps_calibration.parquet"))
    df = df[df["wpe_trend"].notna() & df["recovery_score"].notna()].copy()
    df["biome"] = df["eco_id"].map(BIOME_MAP).fillna("Other")
    print(f"Loaded {len(df):,} pixels")

    # Define WPE context flag: significant positive woody trend
    df["wpe_sig_pos"] = (df["wpe_sig"] == 1) & (df["wpe_trend"] > 0)
    df["wpe_sig_neg"] = (df["wpe_sig"] == 1) & (df["wpe_trend"] < 0)
    df["wpe_nonsig"] = df["wpe_sig"] == 0

    # ── Summary stats ────────────────────────────────────────────────
    print(f"\nWPE significant trend breakdown:")
    print(f"  Sig positive (woody increase): {df['wpe_sig_pos'].sum():,} "
          f"({100*df['wpe_sig_pos'].mean():.1f}%)")
    print(f"  Sig negative (woody decrease): {df['wpe_sig_neg'].sum():,} "
          f"({100*df['wpe_sig_neg'].mean():.1f}%)")
    print(f"  Not significant:               {df['wpe_nonsig'].sum():,} "
          f"({100*df['wpe_nonsig'].mean():.1f}%)")

    # ── Figure 1: Recovery score distributions by WPE status ─────────
    biomes = ["Grassland", "Savanna/Bushveld", "Forest mosaic", "Shrubland/Thicket"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, biome in enumerate(biomes):
        ax = axes[i]
        sub = df[df["biome"] == biome]
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        sig_pos = sub[sub["wpe_sig_pos"]]["recovery_score"]
        sig_neg = sub[sub["wpe_sig_neg"]]["recovery_score"]
        nonsig = sub[sub["wpe_nonsig"]]["recovery_score"]

        bins = np.linspace(0, 100, 40)
        if len(nonsig) > 0:
            ax.hist(nonsig, bins=bins, alpha=0.4, color="#95a5a6", density=True,
                    label=f"No sig WPE (n={len(nonsig):,})")
        if len(sig_pos) > 0:
            ax.hist(sig_pos, bins=bins, alpha=0.5, color="#e74c3c", density=True,
                    label=f"Sig woody+ (n={len(sig_pos):,})")
        if len(sig_neg) > 0:
            ax.hist(sig_neg, bins=bins, alpha=0.5, color="#3498db", density=True,
                    label=f"Sig woody- (n={len(sig_neg):,})")

        # Medians
        if len(sig_pos) > 0:
            ax.axvline(sig_pos.median(), color="#e74c3c", ls="--", lw=1.5)
        if len(nonsig) > 0:
            ax.axvline(nonsig.median(), color="#95a5a6", ls="--", lw=1.5)

        # Mann-Whitney
        if len(sig_pos) > 10 and len(nonsig) > 10:
            u, p = stats.mannwhitneyu(sig_pos, nonsig, alternative="two-sided")
            diff = sig_pos.median() - nonsig.median()
            ax.set_title(f"{biome}\nMedian diff = {diff:+.1f}, p = {p:.2e}",
                         fontsize=10)
        else:
            ax.set_title(biome, fontsize=10)

        ax.set_xlabel("Recovery Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Recovery Score by WPE Woody Trend Status\n"
                 "(Does significant woody increase affect recovery scores?)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out1 = PLOT_DIR / "wpe_context_score_by_biome.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out1}")

    # ── Figure 2: SVH slope vs WPE trend, coloured by biome ─────────
    fig2 = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig2)

    # Panel (0,0-1): Scatter SVH slope vs WPE trend
    ax = fig2.add_subplot(gs[0, :2])
    for biome, color in zip(biomes, ["#2ecc71", "#e67e22", "#3498db", "#9b59b6"]):
        sub = df[df["biome"] == biome]
        ax.scatter(sub["wpe_trend"], sub["svh_slope"],
                   s=1, alpha=0.1, c=color, label=biome)
    ax.set_xlabel("WPE trend (% woody cover change/yr)")
    ax.set_ylabel("SVH slope (m/yr, scaled)")
    ax.set_title("SVH slope vs WPE trend by biome")
    ax.legend(fontsize=8, markerscale=10)
    ax.grid(True, alpha=0.3)
    # Correlation
    valid = df[["wpe_trend", "svh_slope"]].dropna()
    r, p = stats.spearmanr(valid["wpe_trend"], valid["svh_slope"])
    ax.text(0.02, 0.98, f"Spearman r = {r:.3f}, p = {p:.2e}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Panel (0,2): Per-biome correlation summary
    ax2 = fig2.add_subplot(gs[0, 2])
    ax2.axis("off")
    lines = ["Spearman correlation: SVH slope vs WPE trend\n"]
    lines.append(f"{'Biome':<25} {'r':>6} {'p':>10} {'n':>8}")
    lines.append("-" * 52)
    for biome in biomes:
        sub = df[df["biome"] == biome][["wpe_trend", "svh_slope"]].dropna()
        if len(sub) > 10:
            r_b, p_b = stats.spearmanr(sub["wpe_trend"], sub["svh_slope"])
            lines.append(f"{biome:<25} {r_b:>6.3f} {p_b:>10.2e} {len(sub):>8,}")
    ax2.text(0.05, 0.95, "\n".join(lines), transform=ax2.transAxes,
             va="top", fontsize=9, fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Panel (1,0): Grassland-specific: recovery score vs WPE trend
    ax3 = fig2.add_subplot(gs[1, 0])
    grass = df[df["biome"] == "Grassland"]
    sc = ax3.scatter(grass["wpe_trend"], grass["recovery_score"],
                     s=2, alpha=0.15, c=grass["svh_slope"],
                     cmap="RdYlGn", vmin=-0.05, vmax=0.05)
    plt.colorbar(sc, ax=ax3, label="SVH slope (m/yr)", shrink=0.8)
    ax3.set_xlabel("WPE trend (% cover/yr)")
    ax3.set_ylabel("Recovery score")
    ax3.set_title("Grasslands: score vs WPE\n(color = SVH slope)")
    ax3.grid(True, alpha=0.3)

    # Panel (1,1): Savanna-specific
    ax4 = fig2.add_subplot(gs[1, 1])
    sav = df[df["biome"] == "Savanna/Bushveld"]
    sc2 = ax4.scatter(sav["wpe_trend"], sav["recovery_score"],
                      s=2, alpha=0.15, c=sav["svh_slope"],
                      cmap="RdYlGn", vmin=-0.05, vmax=0.05)
    plt.colorbar(sc2, ax=ax4, label="SVH slope (m/yr)", shrink=0.8)
    ax4.set_xlabel("WPE trend (% cover/yr)")
    ax4.set_ylabel("Recovery score")
    ax4.set_title("Savanna/Bushveld: score vs WPE\n(color = SVH slope)")
    ax4.grid(True, alpha=0.3)

    # Panel (1,2): Summary table of proposed context flags
    ax5 = fig2.add_subplot(gs[1, 2])
    ax5.axis("off")

    # Define context flag: sig positive WPE AND positive SVH slope
    # (woody increase driving the structural recovery signal)
    df["woody_context"] = (
        df["wpe_sig_pos"] &
        (df["svh_slope"] > 0)
    )
    lines2 = ["Proposed context flag:\n"
              "  wpe_sig=1 AND wpe_trend>0 AND svh_slope>0\n"
              "  = 'recovery partly driven by woody increase'\n"]
    lines2.append(f"{'Biome':<25} {'flagged':>8} {'total':>8} {'%':>6}")
    lines2.append("-" * 50)
    for biome in biomes:
        sub = df[df["biome"] == biome]
        n_flag = sub["woody_context"].sum()
        n_tot = len(sub)
        pct = 100 * n_flag / n_tot if n_tot > 0 else 0
        lines2.append(f"{biome:<25} {n_flag:>8,} {n_tot:>8,} {pct:>5.1f}%")
    n_all = df["woody_context"].sum()
    lines2.append("-" * 50)
    lines2.append(f"{'TOTAL':<25} {n_all:>8,} {len(df):>8,} "
                  f"{100*n_all/len(df):>5.1f}%")

    # Score comparison
    flagged = df[df["woody_context"]]["recovery_score"]
    unflagged = df[~df["woody_context"]]["recovery_score"]
    lines2.append(f"\nMedian recovery score:")
    lines2.append(f"  Flagged:   {flagged.median():.1f}")
    lines2.append(f"  Unflagged: {unflagged.median():.1f}")

    ax5.text(0.05, 0.95, "\n".join(lines2), transform=ax5.transAxes,
             va="top", fontsize=8, fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig2.suptitle("WPE × SVH Interaction Analysis",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out2 = PLOT_DIR / "wpe_context_interaction.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # ── Figure 3: Per-ecoregion breakdown of flagged pixels ──────────
    ecos = sorted(df["eco_id"].unique())
    ecos = [e for e in ecos if (df["eco_id"] == e).sum() > 50]

    n_cols = 4
    n_rows = (len(ecos) + n_cols - 1) // n_cols
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3.5))
    axes3 = axes3.flatten()

    eco_summary = []
    for i, eco_id in enumerate(ecos):
        ax = axes3[i]
        sub = df[df["eco_id"] == eco_id]
        flagged_e = sub[sub["woody_context"]]
        unflagged_e = sub[~sub["woody_context"]]

        bins = np.linspace(0, 100, 35)
        if len(unflagged_e) > 0:
            ax.hist(unflagged_e["recovery_score"], bins=bins, alpha=0.5,
                    color="#2ecc71", density=True,
                    label=f"No flag (n={len(unflagged_e):,})")
        if len(flagged_e) > 0:
            ax.hist(flagged_e["recovery_score"], bins=bins, alpha=0.5,
                    color="#e74c3c", density=True,
                    label=f"Woody+ (n={len(flagged_e):,})")
            ax.axvline(flagged_e["recovery_score"].median(),
                       color="#e74c3c", ls="--", lw=1.5)
        ax.axvline(unflagged_e["recovery_score"].median() if len(unflagged_e) > 0 else 0,
                   color="#2ecc71", ls="--", lw=1.5)

        name = eco_names.get(eco_id, f"eco{eco_id}")
        short = name[:28] + "..." if len(name) > 28 else name
        biome = BIOME_MAP.get(eco_id, "?")
        pct = 100 * len(flagged_e) / len(sub) if len(sub) > 0 else 0

        if len(flagged_e) > 10 and len(unflagged_e) > 10:
            diff = flagged_e["recovery_score"].median() - unflagged_e["recovery_score"].median()
            ax.set_title(f"{short} ({eco_id}) [{biome}]\n"
                         f"{pct:.0f}% flagged, score diff = {diff:+.1f}",
                         fontsize=8)
        else:
            ax.set_title(f"{short} ({eco_id}) [{biome}]\n{pct:.0f}% flagged",
                         fontsize=8)

        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Recovery Score", fontsize=7)

        eco_summary.append({
            "eco_id": eco_id,
            "eco_name": name,
            "biome": biome,
            "n_total": len(sub),
            "n_flagged": len(flagged_e),
            "pct_flagged": pct,
            "score_flagged": flagged_e["recovery_score"].median() if len(flagged_e) > 0 else np.nan,
            "score_unflagged": unflagged_e["recovery_score"].median() if len(unflagged_e) > 0 else np.nan,
        })

    for j in range(i + 1, len(axes3)):
        axes3[j].set_visible(False)

    fig3.suptitle("Recovery Score: Woody-encroachment-flagged vs Unflagged\n"
                  "(Flag = sig positive WPE trend AND positive SVH slope)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out3 = PLOT_DIR / "wpe_context_per_ecoregion.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out3}")

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n{'='*95}")
    print("WPE CONTEXT FLAG SUMMARY")
    print(f"Flag: wpe_sig_mask=1 AND wpe_trend>0 AND svh_slope>0")
    print(f"{'='*95}")
    print(f"{'eco':>5} {'biome':<20} {'name':<30} {'n':>7} {'flagged':>8} "
          f"{'%':>5} {'sc_flag':>8} {'sc_rest':>8} {'diff':>6}")
    print("-" * 95)
    for r in sorted(eco_summary, key=lambda x: x["eco_id"]):
        name = r["eco_name"][:28]
        biome = r["biome"][:18]
        diff = r["score_flagged"] - r["score_unflagged"] if not np.isnan(r["score_flagged"]) else float("nan")
        print(f"{r['eco_id']:>5} {biome:<20} {name:<30} {r['n_total']:>7,} "
              f"{r['n_flagged']:>8,} {r['pct_flagged']:>4.0f}% "
              f"{r['score_flagged']:>8.1f} {r['score_unflagged']:>8.1f} "
              f"{diff:>+6.1f}")
    print(f"{'='*95}")


if __name__ == "__main__":
    main()
