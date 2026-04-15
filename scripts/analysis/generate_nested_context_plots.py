import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.patches as patches

# Setup
out_dir = r"c:\Users\coach\myfiles\postdoc2\code\plots\narrative"
os.makedirs(out_dir, exist_ok=True)
sns.set_theme(style="white", context="talk", font="sans-serif")
plt.rcParams['font.size'] = 18

# Data (Million Hectares)
area_sa = 122.0
area_active = 12.4
area_abandoned = 3.0
area_recovering = 0.26
area_native = 0.20

# Radius proportionally scales with the square root of Area
def get_radius(area):
    return np.sqrt(area / np.pi)

r_sa = get_radius(area_sa)
r_active = get_radius(area_active)
r_ab = get_radius(area_abandoned)
r_rec = get_radius(area_recovering)
r_nat = get_radius(area_native)

def build_base_axes():
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_aspect('equal')
    ax.axis('off')
    # Wide x-limits to perfectly fit all annotations identically on the right side
    ax.set_xlim(-r_sa * 1.05, r_sa * 1.6) 
    ax.set_ylim(-r_sa * 0.1, r_sa * 2.1)
    return fig, ax

def draw_circle(ax, r, color, alpha=1.0, ec='black', lw=2, zorder=1):
    c = patches.Circle((0, r), r, facecolor=color, alpha=alpha, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(c)
    return c

# ==========================================
# LAYER 1: National Context (SA -> Active -> Abandoned)
# ==========================================
fig, ax = build_base_axes()
draw_circle(ax, r_sa, color='#ecf0f1', ec='#bdc3c7', zorder=1)
draw_circle(ax, r_active, color='#27ae60', alpha=0.9, zorder=2)
draw_circle(ax, r_ab, color='#e67e22', alpha=0.9, zorder=3)

# Annotations (Right aligned anchors)
ax.annotate(f"Total South Africa\n122.0 Mha", xy=(0, r_sa*2), xytext=(r_sa*0.5, r_sa*2),
            arrowprops=dict(arrowstyle="-", color='gray', lw=1.5), va='center', fontsize=22, color='#7f8c8d', fontweight='bold')
            
ax.annotate(f"Active Cropland\n12.4 Mha (~10%)", xy=(0, r_active*2), xytext=(r_sa*0.5, r_active*2 + 0.5),
            arrowprops=dict(arrowstyle="-", color='gray', lw=1.5), va='center', fontsize=22, color='#27ae60', fontweight='bold')
            
ax.annotate(f"Abandoned Agriculture\n3.0 Mha", xy=(0, r_ab*2), xytext=(r_sa*0.5, r_ab*2 + 0.5),
            arrowprops=dict(arrowstyle="-", color='gray', lw=1.5), va='center', fontsize=22, color='#e67e22', fontweight='bold')

plt.title("Layer 1: The National Scale of Abandoned Land", pad=20, fontsize=28, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "context_layer1.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# LAYER 2: Abandoned vs Recovering
# ==========================================
fig, ax = build_base_axes()
draw_circle(ax, r_sa, color='#ecf0f1', ec='#bdc3c7', zorder=1)
draw_circle(ax, r_active, color='#27ae60', alpha=0.2, zorder=2) # Dimmed to focus on inner circles
draw_circle(ax, r_ab, color='#e67e22', alpha=0.9, zorder=3)
draw_circle(ax, r_rec, color='#2980b9', alpha=1.0, zorder=4)

ax.annotate(f"Abandoned Agriculture\n3.0 Mha", xy=(0, r_ab*2), xytext=(r_sa*0.5, r_ab*2 + 0.5),
            arrowprops=dict(arrowstyle="-", color='gray', lw=1.5), va='center', fontsize=22, color='#e67e22', fontweight='bold')
            
# Point to recovering
ax.annotate(f"Recovering\n0.26 Mha (~8.6%)", xy=(r_rec, r_rec), xytext=(r_sa*0.5, r_rec),
            arrowprops=dict(arrowstyle="-", color='gray', lw=1.5), va='center', fontsize=22, color='#2980b9', fontweight='bold')

plt.title("Layer 2: The Fraction Actually Recovering", pad=20, fontsize=28, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "context_layer2.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# LAYER 3: Recovering vs Invaded
# ==========================================
fig, ax = build_base_axes()
draw_circle(ax, r_sa, color='#ecf0f1', ec='#bdc3c7', zorder=1)
draw_circle(ax, r_active, color='#27ae60', alpha=0.1, zorder=2) # Deeply dimmed
draw_circle(ax, r_ab, color='#e67e22', alpha=0.2, zorder=3)     # Dimmed
# The recovering base area acts as the 'Invaded' ring colored Red
draw_circle(ax, r_rec, color='#e74c3c', alpha=1.0, zorder=4)
# The true native area sits strictly inside the recovering circle colored Green
draw_circle(ax, r_nat, color='#27ae60', alpha=1.0, zorder=5)

ax.annotate(f"Recovering (Alien Invaded)\n0.06 Mha (23%)", xy=(r_rec*0.5, r_rec*1.8), xytext=(r_sa*0.5, r_rec*2 + 0.5),
            arrowprops=dict(arrowstyle="-", color='gray', lw=1.5), va='center', fontsize=22, color='#e74c3c', fontweight='bold')
            
ax.annotate(f"Recovering (True Native)\n0.20 Mha (77%)", xy=(r_nat, r_nat), xytext=(r_sa*0.5, r_nat),
            arrowprops=dict(arrowstyle="-", color='gray', lw=1.5), va='center', fontsize=22, color='#27ae60', fontweight='bold')

plt.title("Layer 3: The Composition of 'Recovering' Land", pad=20, fontsize=28, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "context_layer3.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Strictly consistent nested context plots generated successfully.")
