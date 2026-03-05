import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np

model_files = ["QBModel.json", "RBModel.json", "WRModel.json", "TEModel.json"]
positions = ["QB", "RB", "WR", "TE"]
colors = ["orchid", "tomato", "teal", "goldenrod"]

for pos, file, color in zip(positions, model_files, colors):
    model = xgb.XGBRegressor()
    model.load_model(file)

    try:
        importances = model.feature_importances_
        try:
            features = model.feature_names_in_
        except AttributeError:
            features = [f"Feature {i}" for i in range(len(importances))]

        df = pd.DataFrame({'Feature': features, 'Importance': importances})
        df = df.sort_values(by='Importance', ascending=True).tail(10)

        plt.figure(figsize=(10, 8))
        plt.barh(df['Feature'], df['Importance'], color=color, edgecolor='black')

        plt.title(f"{pos} Feature Importance")
        plt.xlabel("Importance (Normalized)")
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{pos}Graph.png")

    except Exception as e:
        print(f"Could not plot {pos}: {e}")

modelRanks = ["qb_predictions.csv", "rb_predictions.csv", "wr_predictions.csv", "te_predictions.csv"]
espnRanks = ["espn_qb_predictions.csv", "espn_rb_predictions.csv", "espn_wr_predictions.csv", "espn_te_predictions.csv"]
actualRanks = ["espn_qb_final.csv", "espn_rb_final.csv", "espn_wr_final.csv", "espn_te_final.csv"]
colors = [["orchid", "tomato"], ["teal", "goldenrod"], ["navy", "crimson"], ["limegreen", "sienna"]]
positions = [["QB", 20], ["RB", 50], ["WR", 60], ["TE", 20]]

for model, espn, actual, colored, positioned in zip(modelRanks, espnRanks, actualRanks, colors, positions):
    position = positioned[0]
    limit = positioned[1]
    color1 = colored[0]
    color2 = colored[1]

    modelFile = pd.read_csv(model).head(limit)
    espnFile = pd.read_csv(espn)
    actualFile = pd.read_csv(actual)

    modelFile["modelRank"] = modelFile.index + 1
    espnFile["espnRank"] = espnFile.index + 1
    actualFile["actualRank"] = actualFile.index + 1

    players = modelFile.merge(espnFile, on="player_name", how="inner")
    players = players.merge(actualFile, on="player_name", how="inner")

    players["modelError"] = players["modelRank"] - players["actualRank"]
    players["espnError"] = players["espnRank"] - players["actualRank"]

    plt.figure(figsize=(8, 6))

    plt.scatter(players["actualRank"], players["modelError"], color=color1, label="Model Error", alpha=0.6,
                edgecolors='w', s=50)
    plt.scatter(players["actualRank"], players["espnError"], color=color2, label="ESPN Error", alpha=0.6,
                edgecolors='w', s=50)

    plt.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Perfect Prediction")

    plt.xlabel("Actual Season Rank")
    plt.ylabel("Prediction Error (Ranks)")
    plt.title(f"{position}: Ranking Error Comparison")

    plt.axhspan(-10, 10, facecolor='gray', alpha=0.1, label="High Accuracy Zone (±10)")

    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{position}ErrorAnalysis.png")

positions = ["QB", "RB", "WR", "TE"]
model_spearman = [0.065, 0.452, 0.438, 0.481]
espn_spearman = [-0.232, 0.692, 0.367, 0.018]

x = np.arange(len(positions))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, model_spearman, width, label='Your Model', color='teal', edgecolor='black')
rects2 = ax.bar(x + width/2, espn_spearman, width, label='ESPN Consensus', color='tomato', edgecolor='black')

ax.set_ylabel('Spearman Correlation (Higher is Better)')
ax.set_title('Ordinal Ranking Accuracy: Model vs. ESPN')
ax.set_xticks(x)
ax.set_xticklabels(positions)
ax.legend()

plt.axhline(0, color='black', linewidth=1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("SpearmanComparison.png")
