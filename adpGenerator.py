from espn_api.football import League
import pandas as pd

# obtain all players
league = League(league_id=1957880705, year=2025)
players = league.free_agents(size=2000)

qbList, rbList, wrList, teList = [], [], [], []

# add players and projected fantasy pts
for player in players:
    player_data = {
        "player_name": player.name,
        "fantasy_pts": player.projected_total_points
    }
    if player.position == "QB":
        qbList.append(player_data)
    elif player.position == "RB":
        rbList.append(player_data)
    elif player.position == "WR":
        wrList.append(player_data)
    elif player.position == "TE":
        teList.append(player_data)

qbDF = pd.DataFrame(qbList)
rbDF = pd.DataFrame(rbList)
wrDF = pd.DataFrame(wrList)
teDF = pd.DataFrame(teList)

# create espn rankings
qbDF.sort_values(by=["fantasy_pts"], inplace=True, ascending=False)
rbDF.sort_values(by=["fantasy_pts"], inplace=True, ascending=False)
wrDF.sort_values(by=["fantasy_pts"], inplace=True, ascending=False)
teDF.sort_values(by=["fantasy_pts"], inplace=True, ascending=False)

qbDF.to_csv("rankings/espn_qb_predictions.csv")
rbDF.to_csv("rankings/espn_rb_predictions.csv")
wrDF.to_csv("rankings/espn_wr_predictions.csv")
teDF.to_csv("rankings/espn_te_predictions.csv")