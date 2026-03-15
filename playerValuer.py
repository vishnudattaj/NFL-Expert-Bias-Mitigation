import pandas as pd

# function identifies overvalued and undervalued players
def process_position(model_file, espn_file):
    model_df = pd.read_csv(model_file)
    espn_df = pd.read_csv(espn_file)

    model_df["model_rank"] = model_df.index + 1
    espn_df["espn_rank"] = espn_df.index + 1

    merged = pd.merge(model_df, espn_df, on="player_name", suffixes=("_model", "_espn"))

    # obtains difference between model and espn projections
    merged["rank_diff"] = merged["espn_rank"] - merged["model_rank"]

    # if model has player ranked at lower number than espn, they're undervalued
    undervalued = merged[merged["rank_diff"] >= 1]
    # opposite is overvalued
    overvalued = merged[merged["rank_diff"] < 0]

    return undervalued[["player_name", "fantasy_pts_model", "fantasy_pts_espn", "model_rank", "espn_rank", "rank_diff"]], overvalued[["player_name", "fantasy_pts_model", "fantasy_pts_espn", "model_rank", "espn_rank", "rank_diff"]]

positions = {
    "qb": ("rankings/qb_predictions.csv", "rankings/espn_qb_predictions.csv"),
    "rb": ("rankings/rb_predictions.csv", "rankings/espn_rb_predictions.csv"),
    "wr": ("rankings/wr_predictions.csv", "rankings/espn_wr_predictions.csv"),
    "te": ("rankings/te_predictions.csv", "rankings/espn_te_predictions.csv")
}

# add to excel book
with pd.ExcelWriter("ranking_comparison.xlsx") as writer:
    for pos, (model_file, espn_file) in positions.items():
        undervalued, overvalued = process_position(model_file, espn_file)
        undervalued.to_excel(writer, sheet_name=f"undervalued_{pos}", index=False)
        overvalued.to_excel(writer, sheet_name=f"overvalued_{pos}", index=False)