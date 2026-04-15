import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import warnings
from adjustText import adjust_text

# Suppress minor warnings for clean execution
warnings.filterwarnings('ignore')

# Set output directory
out_dir = r"c:\Users\coach\myfiles\postdoc2\code\plots\narrative"
os.makedirs(out_dir, exist_ok=True)

# Data from the markdown file
data = [
    ["Highveld grasslands", 81, 826973, 51.9, 149787, 18.1],
    ["Drakensberg grasslands, woodlands and forests", 41, 391506, 49.7, 134278, 34.3],
    ["Central Bushveld", 38, 263016, 53.8, 15476, 5.9],
    ["KwaZulu-Cape coastal forest mosaic", 40, 424697, 48.9, 123320, 29.0],
    ["Kalahari xeric savanna", 97, 35490, 50.6, 2664, 7.5],
    ["Maputaland coastal forest mosaic", 48, 241088, 43.2, 42523, 17.6],
    ["Fynbos", 89, 205112, 49.9, 52569, 25.6],
    ["Succulent Karoo", 90, 101088, 43.3, 11252, 11.1],
    ["Knysna-Amatole montane forests", 101, 16393, 50.3, 5276, 32.2],
    ["Albany thickets", 88, 98706, 50.3, 14897, 15.1],
    ["Zambezian and mopane woodlands", 110, 23767, 50.5, 4977, 20.9],
    ["Eastern Zimbabwe montane forest-grassland mosaic", 16, 156685, 46.8, 80866, 51.6],
    ["Namaqualand-Richtersveld", 102, 9009, 55.5, 10, 0.1],
    ["Maputaland-Pondoland bushland and thickets", 19, 47621, 54.4, 7217, 15.2],
    ["Southern Africa bushveld", 94, 14710, 50.7, 5727, 38.9],
    ["Lowland fynbos and renosterveld", 15, 13132, 39.2, 6256, 47.6],
    ["Montane fynbos and renosterveld", 116, 12282, 25.5, 4122, 33.6],
    ["Nama Karoo", 65, 10, 30.8, 0, 0.0]
]

cols = ["Ecoregion", "ID", "TotalPixels", "RecoveryScore", "InvasivePixels", "PctInvasive"]
df = pd.DataFrame(data, columns=cols)

# Set visual style for publication
sns.set_theme(style="ticks", context="talk", font="sans-serif")
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.title_fontsize'] = 22
plt.rcParams['axes.linewidth'] = 1.8

# 1. Recovery Score by Ecoregion
plt.figure(figsize=(14, 10))
df_sorted = df.sort_values(by="RecoveryScore", ascending=True)

# Create a colormap based on scores
cmap = sns.color_palette("viridis_r", len(df_sorted))
bars = sns.barplot(data=df_sorted, x="RecoveryScore", y="Ecoregion", hue="Ecoregion", palette=cmap, legend=False)
plt.xlim(0, df_sorted['RecoveryScore'].max() * 1.15)
plt.axvline(50, color='#e74c3c', linestyle='--', linewidth=2, label='Natural Median (50)')
sns.despine(left=True, bottom=False)

# Add value labels
for p in bars.patches:
    width = p.get_width()
    if width > 0:
        plt.text(width + 0.5, p.get_y() + p.get_height() / 2, f'{width:.1f}', 
                 ha='left', va='center', fontsize=20, color='black', alpha=0.8)

plt.title("Median Ecological Recovery Score by Ecoregion", pad=20, fontweight='bold')
plt.xlabel("Composite Recovery Score (0-100)", labelpad=15)
plt.ylabel("")
plt.legend(loc='lower left', frameon=True, shadow=True, bbox_to_anchor=(0.02, 0.02))
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "recovery_scores_by_ecoregion.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. Percentage of Invasive Plants by Ecoregion
plt.figure(figsize=(14, 10))
df_sorted_inv = df.sort_values(by="PctInvasive", ascending=True)
cmap_inv = sns.color_palette("Reds", len(df_sorted_inv))
bars_inv = sns.barplot(data=df_sorted_inv, x="PctInvasive", y="Ecoregion", hue="Ecoregion", palette=cmap_inv, legend=False)
plt.xlim(0, df_sorted_inv['PctInvasive'].max() * 1.15)
sns.despine(left=True, bottom=False)

# Add value labels
for p in bars_inv.patches:
    width = p.get_width()
    if width > 0:
        plt.text(width + 0.5, p.get_y() + p.get_height() / 2, f'{width:.1f}%', 
                 ha='left', va='center', fontsize=20, color='black', alpha=0.8)

plt.title("Recovering Pixels Invaded by Alien Plants", pad=20, fontweight='bold')
plt.xlabel("Percentage Invaded (%)", labelpad=15)
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "invasive_plants_by_ecoregion.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. Recovery Score vs % Invasive Scatter Plot
plt.figure(figsize=(14, 9))
scatter = sns.scatterplot(data=df, x="PctInvasive", y="RecoveryScore", size="TotalPixels", sizes=(100, 2500),
                          alpha=0.6, color="#2ecc71", edgecolor='black', linewidth=1)

# Label points cleanly
texts = []
for i in range(df.shape[0]):
    name = df['Ecoregion'][i]
    if df['TotalPixels'][i] > 100000 or df['PctInvasive'][i] > 40 or df['RecoveryScore'][i] < 35 or df['RecoveryScore'][i] > 54:
        if len(name) > 22:
            name = name[:19] + "..."
        # Initial offset so it doesn't start exactly in the center of huge bubbles
        t = plt.text(df['PctInvasive'][i] + 0.5, df['RecoveryScore'][i] + 0.5, name, fontsize=18, 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=4))
        texts.append(t)

# Automatically place texts to avoid overlapping, heavily repelling from bubbles
adjust_text(texts, 
            arrowprops=dict(arrowstyle="-", color='gray', lw=1.5), 
            expand_points=(2.5, 2.5), 
            force_points=(3.0, 3.0),
            force_text=(1.5, 1.5))

plt.axhline(50, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8, label='Natural Median (50)')
sns.despine(bottom=False, left=False)
plt.title("Recovery Score vs. Alien Plant Invasion", pad=20, fontweight='bold')
plt.xlabel("Percentage Invaded by Alien Plants (%)", labelpad=15)
plt.ylabel("Median Composite Recovery Score (0-100)", labelpad=15)

# Adjust legend
handles, labels = scatter.get_legend_handles_labels()
sizes_only_idx = [i for i, handle in enumerate(handles) if type(handle) != plt.Line2D]
# Plot scatter handles manually for size
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Recovering Pixels', shadow=True, markerscale=0.6)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "recovery_vs_invasive.png"), dpi=300, bbox_inches='tight')
plt.close()

# 4. Total Area by Ecoregion (Top 10)
plt.figure(figsize=(14, 10))
df_sorted_area = df.sort_values(by="TotalPixels", ascending=False).head(10)
cmap_area = sns.color_palette("Greens_r", len(df_sorted_area))
bars_area = sns.barplot(data=df_sorted_area, x="TotalPixels", y="Ecoregion", hue="Ecoregion", palette=cmap_area, legend=False)
plt.xlim(0, df_sorted_area['TotalPixels'].max() * 1.15)
sns.despine(left=True, bottom=False)

# Add value labels smartly (inside if long enough, outside if short)
max_width = df_sorted_area['TotalPixels'].max()
for p in bars_area.patches:
    width = p.get_width()
    if width > 0:
        if width > max_width * 0.15:
            # Place inside the bar
            plt.text(width - (max_width * 0.02), p.get_y() + p.get_height() / 2, f'{width/1000:,.0f}k', 
                     ha='right', va='center', fontsize=20, color='white', fontweight='bold')
        else:
            # Place outside the bar
            plt.text(width + (max_width * 0.01), p.get_y() + p.get_height() / 2, f'{width/1000:,.0f}k', 
                     ha='left', va='center', fontsize=20, color='black', alpha=0.8)

plt.title("Top 10 Ecoregions by Recovering Pixel Count", pad=20, fontweight='bold')
plt.xlabel("Number of Recovering Pixels", labelpad=15)
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pixels_by_ecoregion.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Publication quality charts successfully generated.")
