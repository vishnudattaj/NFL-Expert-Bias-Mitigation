import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# function adds custom features to dataframe
def pastFeatures(featureList, excludeList, dataFrame, draft):
    dataFrame = pd.merge(dataFrame.drop(columns=["draft_year"]), draft, on='player_id', how='left')
    # calculate experience
    dataFrame['experience'] = dataFrame['season'] - dataFrame['draft_year']

    # creates lagged features (lags up to 3 years)
    for feature in featureList:
        if feature not in excludeList:
            dataFrame[f"Past-1-{feature}"] = dataFrame.groupby('player_id')[feature].shift(1)
            dataFrame[f"Past-2-{feature}"] = dataFrame.groupby('player_id')[feature].shift(2)
            dataFrame[f"Past-3-{feature}"] = dataFrame.groupby('player_id')[feature].shift(3)

    # calculate career max stats
    dataFrame['career_max_receiving_yards'] = dataFrame.groupby('player_id')['Past-1-receiving_yards'].cummax()
    dataFrame['career_max_rushing_yards'] = dataFrame.groupby('player_id')['Past-1-rushing_yards'].cummax()
    dataFrame['career_max_passing_yards'] = dataFrame.groupby('player_id')['Past-1-passing_yards'].cummax()

    # drop columns that are all na
    dataFrame.dropna(axis=1, how='all', inplace=True)
    # replace individual na with 0
    dataFrame.fillna(0, inplace=True)

    return dataFrame[(dataFrame["season"] > 2012)].reset_index(drop=True)

# function extracts current data
def currentDataExtractor(dataFrame, excludeList, model, trainColumns, playerType):
    df_2024 = dataFrame[dataFrame["season"] == 2024].copy()

    # recreate experience and age
    if 'experience' in df_2024.columns:
        df_2024['experience'] += 1
    if 'age' in df_2024.columns:
        df_2024['age'] += 1

    # remake lags to predict 2025 data
    for feature in df_2024.columns:
        if feature not in excludeList and "Past" not in feature:
            if f"Past-2-{feature}" in df_2024.columns:
                df_2024[f"Past-3-{feature}"] = df_2024[f"Past-2-{feature}"]

            if f"Past-1-{feature}" in df_2024.columns:
                df_2024[f"Past-2-{feature}"] = df_2024[f"Past-1-{feature}"]

            df_2024[f"Past-1-{feature}"] = df_2024[feature]

    # calculate career max stats
    df_2024['career_max_receiving_yards'] = df_2024.groupby('player_id')['Past-1-receiving_yards'].cummax()
    df_2024['career_max_rushing_yards'] = df_2024.groupby('player_id')['Past-1-rushing_yards'].cummax()
    df_2024['career_max_passing_yards'] = df_2024.groupby('player_id')['Past-1-passing_yards'].cummax()

    # predict using model
    X = df_2024[model.feature_names_in_]
    y_pred = model.predict(X)

    pred_df = pd.DataFrame(
        y_pred,
        columns=trainColumns
    )

    pred_df["player_id"] = df_2024["player_id"].values
    pred_df["player_name"] = df_2024["player_name"].values
    # calculate fantasy pts
    if playerType == "QB":
        pred_df["fantasy_pts"] = (pred_df["passing_yards"] / 25) + (pred_df["pass_touchdown"] * 4) + (pred_df["rushing_yards"] / 10) + (pred_df["rush_touchdown"] * 6) - (pred_df["interception"] * 2) - (pred_df["fumble"] * 2)
    else:
        pred_df["fantasy_pts"] = (pred_df["rushing_yards"] / 10) + (pred_df["rush_touchdown"] * 6) + (pred_df["receiving_yards"] / 10) + (pred_df["receiving_touchdown"] * 6) + (pred_df["receptions"] * 1) - (pred_df["fumble"] * 2)
    # sort by fantasy pts
    pred_df = pred_df.sort_values(by="fantasy_pts", ascending=False).reset_index(drop=True)
    pred_df = pred_df[["player_id", "player_name", "fantasy_pts"] + trainColumns]
    return pred_df

# function trains models
def trainModel(trainingDF, trainColumns, pos):
    feature_names = [col for col in trainingDF.columns if any(x in col for x in ["Past", "career_max", "experience"])]

    # testing data is 2024, train using every year but 2024
    train_df = trainingDF[trainingDF["season"] < 2024]
    test_df = trainingDF[trainingDF["season"] == 2024]

    X_train = train_df[feature_names]
    y_train = train_df[trainColumns]

    X_test = test_df[feature_names]
    y_test = test_df[trainColumns]

    # set parameters
    model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            tree_method="hist",
            multi_strategy="multi_output_tree",
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=1,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
    )

    # train model
    model.fit(X_train, y_train, verbose=1)

    predictions = model.predict(X_test)

    # evaluate R2
    print(f"\n--- Evaluation for {pos} ---")
    for i, col in enumerate(trainColumns):
        r2 = r2_score(y_test.iloc[:, i], predictions[:, i])
        print(f"{col} -> R2: {r2:.2f}")

    # save model
    model.save_model(f"models/{pos}Model.json")

    return model

# read dataframe
yearlyPlayer = pd.read_csv('yearly_player_stats_offense.csv')
yearlyPlayer = yearlyPlayer[(yearlyPlayer['season_type'] == "REG")]
yearlyPlayer.drop(["season_type", "games_played_season", "games_played_career"], axis=1, inplace=True)

yearlyFeatures = yearlyPlayer.columns.tolist()

# features we dont want to evaluate model off
excludeFeatures = ["player_id", "player_name", "position", "birth_year", "draft_round", "draft_pick", "draft_ovr", "height", "weight", "college", "season", "team", "conference", "division", "career_max_receiving_yards", "td_per_target", 'age_at_rookie']
draft_year_df = yearlyPlayer[['player_id', 'draft_year']].drop_duplicates()

yearlyPlayer = pastFeatures(yearlyFeatures, excludeFeatures, yearlyPlayer, draft_year_df)

# split by position
yearlyQB = yearlyPlayer[(yearlyPlayer['position'] == "QB")].reset_index(drop=True)
yearlyRB = yearlyPlayer[(yearlyPlayer['position'] == "RB")].reset_index(drop=True)
yearlyWR = yearlyPlayer[(yearlyPlayer['position'] == "WR")].reset_index(drop=True)
yearlyTE = yearlyPlayer[(yearlyPlayer['position'] == "TE")].reset_index(drop=True)

# train model and evaluate, save rankings into csv
qbDF = currentDataExtractor(yearlyQB, excludeFeatures, trainModel(yearlyQB, ["passing_yards", "rushing_yards", "rush_touchdown", "pass_touchdown", "interception", "fumble"], "QB"), ["passing_yards", "rushing_yards", "rush_touchdown", "pass_touchdown", "interception", "fumble"], "QB")
qbDF.drop(columns=["player_id"], inplace=True)
qbDF.sort_values(by='fantasy_pts', ascending=False).reset_index(drop=True).to_csv("rankings/qb_predictions.csv")

# train model and evaluate, save rankings into csv
rbDF = currentDataExtractor(yearlyRB, excludeFeatures, trainModel(yearlyRB, ["receiving_yards", "rushing_yards", "rush_touchdown", "receptions", "receiving_touchdown", "fumble"], "RB"), ["receiving_yards", "rushing_yards", "rush_touchdown", "receptions", "receiving_touchdown", "fumble"], "RB")
rbDF.drop(columns=["player_id"], inplace=True)
rbDF.sort_values(by='fantasy_pts', ascending=False).reset_index(drop=True).to_csv("rankings/rb_predictions.csv")

# train model and evaluate, save rankings into csv
wrDF = currentDataExtractor(yearlyWR, excludeFeatures, trainModel(yearlyWR, ["receiving_yards", "rushing_yards", "rush_touchdown", "receptions", "receiving_touchdown", "fumble"], "WR"), ["receiving_yards", "rushing_yards", "rush_touchdown", "receptions", "receiving_touchdown", "fumble"], "WR")
wrDF.drop(columns=["player_id"], inplace=True)
wrDF.sort_values(by='fantasy_pts', ascending=False).reset_index(drop=True).to_csv("rankings/wr_predictions.csv")

# train model and evaluate, save rankings into csv
teDF = currentDataExtractor(yearlyTE, excludeFeatures, trainModel(yearlyTE, ["receiving_yards", "rushing_yards", "rush_touchdown", "receptions", "receiving_touchdown", "fumble"], "TE"), ["receiving_yards", "rushing_yards", "rush_touchdown", "receptions", "receiving_touchdown", "fumble"], "TE")
teDF.drop(columns=["player_id"], inplace=True)
teDF.sort_values(by='fantasy_pts', ascending=False).reset_index(drop=True).to_csv("rankings/te_predictions.csv")
