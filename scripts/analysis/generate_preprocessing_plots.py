import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set output directory
out_dir = r"c:\Users\coach\myfiles\postdoc2\code\plots\narrative"
os.makedirs(out_dir, exist_ok=True)

# Set visual style for publication
sns.set_theme(style="ticks", context="talk", font="sans-serif")
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['axes.linewidth'] = 1.8

# 1. Chart breaking down the percentage of data removed (from the ~33.4M pixels)
# We group the non-secondary natural (Class 2) SANLC pixels that were filtered out.
# Total is 33.41M. Retained = 27.86M. Removed = 5.55M.
removal_data = {
    'Category': [
        'Cultivated / Re-established Ag', 
        'Mines, Quarries & Built-up', 
        'Forestry / Plantations', 
        'Bare / Degraded'
    ],
    'Pixels': [3592319, 1377113, 417698, 155836] # SANLC 5, SANLC 1+4, SANLC 7, SANLC 3+6
}
df_removal = pd.DataFrame(removal_data)
# Calculate percentage of total removed data (which is 5.54M pixels)
total_removed = df_removal['Pixels'].sum()
df_removal['Percentage'] = (df_removal['Pixels'] / total_removed) * 100

plt.figure(figsize=(12, 8))
bars_rem = sns.barplot(data=df_removal, x='Percentage', y='Category', palette='magma_r')
sns.despine(left=True, bottom=False)

# Add value annotations
for p in bars_rem.patches:
    width = p.get_width()
    plt.text(width + 1.5, p.get_y() + p.get_height() / 2, f'{width:.1f}%', 
             ha='left', va='center', fontsize=22, color='black', alpha=0.9)

plt.xlim(0, 80) # leave room for labels
plt.title("Breakdown of Excluded Data by Transformed Land Cover", pad=20, fontweight='bold')
plt.xlabel("Percentage of Total Removed Pixels (%)", labelpad=15)
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "data_removal_breakdown.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. Area of active agriculture vs abandoned agriculture
# Active Ag in SA is roughly 14 million hectares, which is 140,000 sq km (Reference: DAFF / DALRRD Abstract of Agricultural Statistics). 
# Abandoned Ag is 33.41M pixels * 900 sq meters = 30,073 sq km.
area_data = {
    'Land Use': ['Active Agriculture\n(Current)', 'Abandoned Agriculture\n(Historic)'],
    'Area': [140000, 30073]
}
df_area = pd.DataFrame(area_data)

plt.figure(figsize=(10, 8))
bars_area = sns.barplot(data=df_area, x='Land Use', y='Area', palette=['#27ae60', '#e67e22'])
sns.despine(left=False, bottom=True)

# Add value annotations inside/above bars
for i, p in enumerate(bars_area.patches):
    height = p.get_height()
    if i == 0:
        label = f'{int(height):,} sq km'
    else:
        pct = (height / 140000) * 100
        label = f'{int(height):,} sq km\n({pct:.1f}%)'
        
    plt.text(p.get_x() + p.get_width() / 2, height + 2000, label, 
             ha='center', va='bottom', fontsize=24, color='black', fontweight='bold')

plt.ylim(0, 160000)
plt.title("Active vs. Abandoned Agriculture Area\n(Active Ref: DAFF Agricultural Statistics)", pad=20, fontweight='bold', fontsize=24)
plt.ylabel("Area in South Africa (sq km)", labelpad=15)
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "area_active_vs_abandoned.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Preprocessing charts successfully generated.")
