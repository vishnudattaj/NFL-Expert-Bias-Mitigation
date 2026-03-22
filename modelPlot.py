import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np
import os

# better quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

model_files = ["models/QBModel.json", "models/RBModel.json", "models/WRModel.json", "models/TEModel.json"]
positions = ["QB", "RB", "WR", "TE"]
colors = ["orchid", "tomato", "teal", "goldenrod"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
axes_flat = axes.flatten()

for i, (pos, file, color) in enumerate(zip(positions, model_files, colors)):
    ax = axes_flat[i]
    model = xgb.XGBRegressor()

    try:
        model.load_model(file)
        importances = model.feature_importances_

        try:
            features = model.feature_names_in_
        except AttributeError:
            features = [f"Feature {i}" for i in range(len(importances))]

        df = pd.DataFrame({'Feature': features, 'Importance': importances})
        # Sort and take top 10
        df = df.sort_values(by='Importance', ascending=True).tail(10)

        # Plotting on the specific subplot axis
        ax.barh(df['Feature'], df['Importance'], color=color, edgecolor='black')

        # Styling each panel
        ax.set_title(f"{pos} Feature Importance", fontsize=16, fontweight='bold')
        ax.set_xlabel("Importance (Normalized)", fontsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    except Exception as e:
        print(f"Could not plot {pos}: {e}")
        ax.text(0.5, 0.5, f"Error loading {pos}", ha='center')

plt.savefig("graphs/Combined_Position_Importance.svg", bbox_inches='tight', facecolor='white')

# add spearman data
positions = ["QB", "RB", "WR", "TE"]
model_spearman = [0.039, 0.532, 0.363, 0.247]
espn_spearman = [-0.232, 0.692, 0.367, 0.018]

x = np.arange(len(positions))
width = 0.35

# create plots
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, model_spearman, width, label='Model', color='teal', edgecolor='black')
rects2 = ax.bar(x + width/2, espn_spearman, width, label='ESPN Consensus', color='tomato', edgecolor='black')

ax.set_ylabel('Spearman Correlation (Higher is Better)')
ax.set_title('Ordinal Ranking Accuracy: Model vs. ESPN')
ax.set_xticks(x)
ax.set_xticklabels(positions)
ax.legend()

plt.axhline(0, color='black', linewidth=1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("graphs/SpearmanComparison.png")

# spearman data accounting for injuries
model_spearman = [0.039, 0.597, 0.326, 0.247]
espn_spearman = [-0.232, 0.696, 0.367, 0.018]

x = np.arange(len(positions))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, model_spearman, width, label='Model', color='teal', edgecolor='black')
rects2 = ax.bar(x + width/2, espn_spearman, width, label='ESPN Consensus', color='tomato', edgecolor='black')

ax.set_ylabel('Spearman Correlation (Higher is Better)')
ax.set_title('Injury Accounted Ordinal Ranking Accuracy: Model vs. ESPN')
ax.set_xticks(x)
ax.set_xticklabels(positions)
ax.legend()

plt.axhline(0, color='black', linewidth=1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("graphs/ErrorAccountedSpearmanComparison.png")

# mae data
positions = ["QB", "RB", "WR", "TE"]
model_mae = [10.105, 30.930, 37.208, 13.579]
espn_mae = [12.250, 17.420, 22.983, 9.450]

x = np.arange(len(positions))
width = 0.35

# Plot MAE Comparison
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, model_mae, width, label='Model', color='teal', edgecolor='black')
plt.bar(x + width/2, espn_mae, width, label='ESPN Consensus', color='tomato', edgecolor='black')

plt.ylabel('Mean Absolute Error (Lower is Better)')
plt.title('Absolute Error Comparison: Model vs. ESPN')
plt.xticks(x, positions)
plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

ymin, ymax = plt.gca().get_ylim()

plt.savefig("graphs/MAEComparison.png")

# MAE data accounting for injuries
model_mae = [10.105, 26.233, 27.712, 13.579]
espn_mae = [12.250, 14.740, 22.983, 9.450]

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, model_mae, width, label='Model', color='teal', edgecolor='black')
plt.bar(x + width/2, espn_mae, width, label='ESPN Consensus', color='tomato', edgecolor='black')

plt.ylabel('Mean Absolute Error (Lower is Better)')
plt.title('Injury Accounted Absolute Error Comparison: Model vs. ESPN')
plt.xticks(x, positions)
plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, ymax)
plt.yticks(range(0, int(ymax)+1, 5))
plt.tight_layout()
plt.savefig("graphs/ErrorAccountedMAEComparison.png")
