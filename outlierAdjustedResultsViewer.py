import pandas as pd
from scipy.stats import spearmanr

def get_metrics(projDF, actualDF, topN):
    zeroGamesPlayed = ["Joe Mixon", "Ezekiel Elliott", "Gus Edwards", "Cordarrelle Patterson", "Brandon Aiyuk", "Tank Dell", "Diontae Johnson"]
    projDF = projDF[~projDF["player_name"].isin(zeroGamesPlayed)].head(topN)[["player_name"]].copy()
    actualDF = actualDF[["player_name"]].copy()

    projDF["modelRank"] = projDF.index + 1
    actualDF["actualRank"] = actualDF.index + 1

    players = projDF.merge(actualDF, on="player_name", how="inner")
    players["rankDiff"] = players["modelRank"] - players["actualRank"]
    players["absDiff"] = players["rankDiff"].abs()

    result = spearmanr(players["modelRank"], players["actualRank"])

    return players["absDiff"].mean(), result.statistic

positions = [["qb", 20], ["rb", 50], ["wr", 60], ["te", 20]]

for pos_name, limit in positions:
    projDF = pd.read_csv(f'rankings/{pos_name}_predictions.csv')
    espnDF = pd.read_csv(f'rankings/espn_{pos_name}_predictions.csv')
    actualDF = pd.read_csv(f'rankings/espn_{pos_name}_final.csv')

    if pos_name == "te":
        projDF.replace("Kyle Pitts", "Kyle Pitts Sr.", inplace=True)

    modelMAE, modelRho = get_metrics(projDF, actualDF, limit)
    espnMAE, espnRho = get_metrics(espnDF, actualDF, limit)

    print(f"--- {pos_name.upper()} ---")
    print(f"MAE: Model: {modelMAE:.3f} | ESPN: {espnMAE:.3f}")
    print(f"Spearman Correlation: Model: {modelRho:.3f} | ESPN: {espnRho:.3f}")
    print()
