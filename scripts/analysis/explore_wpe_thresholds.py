"""
Exploratory analysis: WPE distributions by NIAPS status per ecoregion.
Derives ecoregion-specific woody encroachment thresholds.

Input: data/wpe_niaps_calibration.parquet
Output: plots/wpe_calibration_overview.png
        plots/wpe_roc_per_ecoregion.png
        data/wpe_thresholds.csv
"""
import sys
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

ECO_CSV = DATA_DIR / "ecoregion_sds.csv"


def load_eco_names():
    df = pd.read_csv(ECO_CSV, usecols=["ECO_ID", "ECO_NAME"]).drop_duplicates("ECO_ID")
    return dict(zip(df["ECO_ID"].astype(int), df["ECO_NAME"]))


def main():
    eco_names = load_eco_names()

    # Load calibration data
    df = pd.read_parquet(str(DATA_DIR / "wpe_niaps_calibration.parquet"))
    df = df[df["wpe_trend"].notna()].copy()
    print(f"Loaded {len(df):,} pixels with WPE data")
    print(f"  NIAPS=1: {(df['niaps']==1).sum():,}")
    print(f"  NIAPS=0: {(df['niaps']==0).sum():,}")

    ecos = sorted(df["eco_id"].unique())
    # Drop eco65 (only 10 pixels)
    ecos = [e for e in ecos if (df["eco_id"] == e).sum() > 100]
    n_eco = len(ecos)

    # ── Figure 1: Distribution overview ──────────────────────────────
    n_cols = 4
    n_rows = (n_eco + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3.5))
    axes = axes.flatten()

    threshold_results = []

    for i, eco_id in enumerate(ecos):
        ax = axes[i]
        sub = df[df["eco_id"] == eco_id]
        inv = sub[sub["niaps"] == 1]["wpe_trend"]
        nat = sub[sub["niaps"] == 0]["wpe_trend"]

        # Histograms
        bins = np.linspace(sub["wpe_trend"].quantile(0.01),
                           sub["wpe_trend"].quantile(0.99), 50)
        ax.hist(nat, bins=bins, alpha=0.5, color="#2ecc71", density=True,
                label=f"NIAPS=0 (n={len(nat):,})")
        if len(inv) > 10:
            ax.hist(inv, bins=bins, alpha=0.5, color="#e74c3c", density=True,
                    label=f"NIAPS=1 (n={len(inv):,})")

        # Medians
        ax.axvline(nat.median(), color="#2ecc71", ls="--", lw=1.5)
        if len(inv) > 10:
            ax.axvline(inv.median(), color="#e74c3c", ls="--", lw=1.5)

        # Mann-Whitney U test
        if len(inv) > 10:
            u_stat, u_p = stats.mannwhitneyu(inv, nat, alternative="greater")
            effect = inv.median() - nat.median()
        else:
            u_p = float("nan")
            effect = float("nan")

        name = eco_names.get(eco_id, f"eco{eco_id}")
        short_name = name[:30] + "..." if len(name) > 30 else name
        ax.set_title(f"{short_name} ({eco_id})\n"
                     f"diff={effect:.3f}, p={u_p:.2e}", fontsize=8)
        ax.set_xlabel("WPE trend (% cover/yr)", fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("WPE Trend Distribution: NIAPS-invaded vs Non-invaded\n"
                 "(dashed lines = medians)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out1 = PLOT_DIR / "wpe_calibration_overview.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out1}")

    # ── Figure 2: ROC curves per ecoregion ───────────────────────────
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3.5))
    axes2 = axes2.flatten()

    for i, eco_id in enumerate(ecos):
        ax = axes2[i]
        sub = df[df["eco_id"] == eco_id]
        y_true = sub["niaps"].values
        y_score = sub["wpe_trend"].values

        if y_true.sum() < 10 or (y_true == 0).sum() < 10:
            ax.text(0.5, 0.5, "Too few NIAPS\nfor ROC",
                    ha="center", va="center", transform=ax.transAxes)
            threshold_results.append({
                "eco_id": eco_id,
                "eco_name": eco_names.get(eco_id, ""),
                "n_total": len(sub), "n_inv": int(y_true.sum()),
                "auc": float("nan"), "youden_thresh": float("nan"),
                "p90_thresh": float("nan"),
                "inv_median": float("nan"), "nat_median": float("nan"),
            })
            continue

        # ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        ax.plot(fpr, tpr, color="#3498db", lw=2)
        ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)

        # Youden's J = max(TPR - FPR)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        youden_thresh = thresholds[best_idx]
        ax.scatter(fpr[best_idx], tpr[best_idx], c="#e74c3c", s=60, zorder=5,
                   label=f"Youden={youden_thresh:.2f}")

        # 90th percentile of non-invaded as alternative threshold
        nat_vals = sub[sub["niaps"] == 0]["wpe_trend"]
        inv_vals = sub[sub["niaps"] == 1]["wpe_trend"]
        p90_thresh = nat_vals.quantile(0.90)

        name = eco_names.get(eco_id, f"eco{eco_id}")
        short_name = name[:30] + "..." if len(name) > 30 else name
        ax.set_title(f"{short_name} ({eco_id})\n"
                     f"AUC={auc:.3f}, Youden={youden_thresh:.2f}, "
                     f"P90={p90_thresh:.2f}", fontsize=8)
        ax.set_xlabel("FPR", fontsize=7)
        ax.set_ylabel("TPR", fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        threshold_results.append({
            "eco_id": eco_id,
            "eco_name": eco_names.get(eco_id, ""),
            "n_total": len(sub),
            "n_inv": int(y_true.sum()),
            "inv_pct": 100 * y_true.sum() / len(sub),
            "auc": auc,
            "youden_thresh": youden_thresh,
            "p90_thresh": p90_thresh,
            "inv_median": inv_vals.median(),
            "nat_median": nat_vals.median(),
            "median_diff": inv_vals.median() - nat_vals.median(),
        })

    for j in range(i + 1, len(axes2)):
        axes2[j].set_visible(False)

    fig2.suptitle("ROC Curves: WPE Trend predicting NIAPS invasion\n"
                  "(red dot = Youden's optimal threshold)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out2 = PLOT_DIR / "wpe_roc_per_ecoregion.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # ── Figure 3: WPE trend vs sig_mask interaction ──────────────────
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3.5))
    axes3 = axes3.flatten()

    for i, eco_id in enumerate(ecos):
        ax = axes3[i]
        sub = df[df["eco_id"] == eco_id]

        sig = sub["wpe_sig"] == 1
        nosig = sub["wpe_sig"] == 0
        inv = sub["niaps"] == 1

        # Contingency: significant woody trend AND invaded
        n_sig_inv = (sig & inv).sum()
        n_sig_notinv = (sig & ~inv).sum()
        n_nosig_inv = (nosig & inv).sum()
        n_nosig_notinv = (nosig & ~inv).sum()

        # Stacked bar
        categories = ["Sig+Inv", "Sig+NotInv", "NoSig+Inv", "NoSig+NotInv"]
        counts = [n_sig_inv, n_sig_notinv, n_nosig_inv, n_nosig_notinv]
        colors = ["#e74c3c", "#f39c12", "#95a5a6", "#2ecc71"]
        ax.bar(categories, counts, color=colors)

        # Add percentages
        total = len(sub)
        for j_bar, (cat, cnt) in enumerate(zip(categories, counts)):
            pct = 100 * cnt / total if total > 0 else 0
            ax.text(j_bar, cnt + total * 0.01, f"{pct:.0f}%",
                    ha="center", fontsize=6)

        name = eco_names.get(eco_id, f"eco{eco_id}")
        short_name = name[:25] + "..." if len(name) > 25 else name
        ax.set_title(f"{short_name} ({eco_id})", fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.grid(True, alpha=0.3, axis="y")

    for j in range(i + 1, len(axes3)):
        axes3[j].set_visible(False)

    fig3.suptitle("WPE Significant Trend × NIAPS Invasion\n"
                  "(how well does WPE sig_mask alone separate invaded pixels?)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out3 = PLOT_DIR / "wpe_sig_niaps_interaction.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out3}")

    # ── Summary table ────────────────────────────────────────────────
    result_df = pd.DataFrame(threshold_results)
    result_df = result_df.sort_values("eco_id")
    out_csv = DATA_DIR / "wpe_thresholds.csv"
    result_df.to_csv(str(out_csv), index=False, float_format="%.4f")
    print(f"\nSaved: {out_csv}")

    print(f"\n{'='*100}")
    print("WPE THRESHOLD CALIBRATION SUMMARY")
    print(f"{'='*100}")
    print(f"{'eco':>5} {'name':<35} {'n':>8} {'inv%':>6} {'AUC':>6} "
          f"{'Youden':>8} {'P90':>8} {'med_inv':>8} {'med_nat':>8} {'diff':>6}")
    print("-" * 100)
    for _, r in result_df.iterrows():
        name = r["eco_name"][:32]
        print(f"{int(r['eco_id']):>5} {name:<35} {int(r['n_total']):>8,} "
              f"{r['inv_pct']:>5.1f}% {r['auc']:>6.3f} "
              f"{r['youden_thresh']:>8.3f} {r['p90_thresh']:>8.3f} "
              f"{r['inv_median']:>8.3f} {r['nat_median']:>8.3f} "
              f"{r['median_diff']:>6.3f}")
    print(f"{'='*100}")

    # Key findings
    print("\nKey findings:")
    good_auc = result_df[result_df["auc"] > 0.6]
    poor_auc = result_df[result_df["auc"] <= 0.6]
    print(f"  Ecoregions with AUC > 0.6 (WPE discriminates well): "
          f"{len(good_auc)} — {list(good_auc['eco_id'].astype(int))}")
    print(f"  Ecoregions with AUC <= 0.6 (WPE discriminates poorly): "
          f"{len(poor_auc)} — {list(poor_auc['eco_id'].astype(int))}")

    # Check overall: combined ROC
    y_all = df["niaps"].values
    x_all = df["wpe_trend"].values
    valid = ~np.isnan(x_all)
    auc_all = roc_auc_score(y_all[valid], x_all[valid])
    print(f"\n  Overall AUC (all ecoregions pooled): {auc_all:.3f}")


if __name__ == "__main__":
    main()
