import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np

# import ML models
model_files = ["models/QBModel.json", "models/RBModel.json", "models/WRModel.json", "models/TEModel.json"]
positions = ["QB", "RB", "WR", "TE"]
colors = ["orchid", "tomato", "teal", "goldenrod"]

# assign color and position to model file
for pos, file, color in zip(positions, model_files, colors):
    model = xgb.XGBRegressor()
    model.load_model(file)

    try:
        # obtain importance of features
        importances = model.feature_importances_
        # obtain list of features
        try:
            features = model.feature_names_in_
        except AttributeError:
            features = [f"Feature {i}" for i in range(len(importances))]

        # create dataframe with 10 most important features
        df = pd.DataFrame({'Feature': features, 'Importance': importances})
        df = df.sort_values(by='Importance', ascending=True).tail(10)

        # create plot
        plt.figure(figsize=(10, 8))
        plt.barh(df['Feature'], df['Importance'], color=color, edgecolor='black')

        plt.title(f"{pos} Feature Importance")
        plt.xlabel("Importance (Normalized)")
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"graphs/{pos}Graph.png")

    except Exception as e:
        print(f"Could not plot {pos}: {e}")

# import rankings
modelRanks = ["rankings/qb_predictions.csv", "rankings/rb_predictions.csv", "rankings/wr_predictions.csv",
              "rankings/te_predictions.csv"]
espnRanks = ["rankings/espn_qb_predictions.csv", "rankings/espn_rb_predictions.csv", "rankings/espn_wr_predictions.csv",
             "rankings/espn_te_predictions.csv"]
actualRanks = ["rankings/espn_qb_final.csv", "rankings/espn_rb_final.csv", "rankings/espn_wr_final.csv",
               "rankings/espn_te_final.csv"]
# positions contains position and limiter value
positions = [["QB", 20], ["RB", 50], ["WR", 60], ["TE", 20]]

# zip rankings with colors and positional values
for model, espn, actual, positioned in zip(modelRanks, espnRanks, actualRanks, positions):
    position = positioned[0]
    limit = positioned[1]
    color1 = "teal"
    color2 = "tomato"

    modelFile = pd.read_csv(model).head(limit)
    espnFile = pd.read_csv(espn)
    actualFile = pd.read_csv(actual)

    # adding one to index makes rank begin from 1
    modelFile["modelRank"] = modelFile.index + 1
    espnFile["espnRank"] = espnFile.index + 1
    actualFile["actualRank"] = actualFile.index + 1

    # only compare players in all three files
    players = modelFile.merge(espnFile, on="player_name", how="inner")
    players = players.merge(actualFile, on="player_name", how="inner")

    # find error
    players["modelError"] = players["modelRank"] - players["actualRank"]
    players["espnError"] = players["espnRank"] - players["actualRank"]

    # Calculate counts within +/- 10
    model_hits = len(players[abs(players["modelError"]) <= 10])
    espn_hits = len(players[abs(players["espnError"]) <= 10])
    total_players = len(players)

    # create plots
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

    # Add text box for summary statistics
    stats_text = (f"Model Hits (±10): {model_hits}/{total_players}\n"
                  f"ESPN Hits (±10): {espn_hits}/{total_players}")
    plt.text(0.75, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"graphs/{position}ErrorAnalysis.png")

# add spearman data
positions = ["QB", "RB", "WR", "TE"]
model_spearman = [0.065, 0.452, 0.438, 0.481]
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
model_spearman = [0.065, 0.525, 0.446, 0.481]
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
model_mae = [11.400, 43.302, 40.231, 14.000]
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
model_mae = [11.400, 26.721, 24.519, 14.000]
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
plt.yticks(range(0, int(ymax)+1, 10))
plt.tight_layout()
plt.savefig("graphs/ErrorAccountedMAEComparison.png")
