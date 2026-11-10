import pandas as pd
import numpy as np
import time
# from sklearn.preprocessing import StandardScaler
from nba_api.stats.endpoints import leaguedashteamstats
from xgboost import XGBRanker
from lightgbm import LGBMRanker

# === 1. Retrieve NBA team statistics from regular season games（Base + Advanced) ===
def fetch_season_data(season):
    time.sleep(1)
    base_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star='Regular Season',
        measure_type_detailed_defense='Base',
        per_mode_detailed='PerGame',
        timeout=60
    )
    base_df = base_stats.get_data_frames()[0]
    time.sleep(1)
    adv_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star='Regular Season',
        measure_type_detailed_defense='Advanced',
        per_mode_detailed='PerGame'
    )
    adv_df = adv_stats.get_data_frames()[0]
    merged = pd.merge(base_df, adv_df, on='TEAM_ID', suffixes=('_base', '_adv'))
    merged['SEASON'] = season
    return merged

# === 2. Fetch 2000–2025 data ===
seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(2000, 2025)]
all_data = pd.concat([fetch_season_data(season) for season in seasons], ignore_index=True)

# === 3. Column selection and feature engineering ===
columns_to_keep = [
    'TEAM_ID', 'TEAM_NAME_base', 'SEASON', 'FGA', 'W_PCT_base',
    'FG_PCT', 'FG3A', 'FG3_PCT', 'FTA', 'FT_PCT', 'OREB',
    'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'PTS',
    'OFF_RATING', 'EFG_PCT', 'TS_PCT',
    'DEF_RATING', 'REB_PCT', 'DREB_PCT',
    'NET_RATING', 'PIE',
    'AST_RATIO', 'AST_PCT', 'AST_TO'
]
data = all_data[columns_to_keep].copy()

# Add derived features
data['PTS_per_FGA'] = data['PTS'] / data['FGA']
data['AST_TOV_ratio'] = data['AST'] / (data['TOV'] + 1e-5)

# === 4. Champion Mark ===
champions = {
    # '1996-97': 'Chicago Bulls', '1997-98': 'Chicago Bulls',
    # '1998-99': 'San Antonio Spurs', '1999-00': 'Los Angeles Lakers',
    '2000-01': 'Los Angeles Lakers', '2001-02': 'Los Angeles Lakers',
    '2002-03': 'San Antonio Spurs', '2003-04': 'Detroit Pistons',
    '2004-05': 'San Antonio Spurs', '2005-06': 'Miami Heat',
    '2006-07': 'San Antonio Spurs', '2007-08': 'Boston Celtics',
    '2008-09': 'Los Angeles Lakers', '2009-10': 'Los Angeles Lakers',
    '2010-11': 'Dallas Mavericks', '2011-12': 'Miami Heat',
    '2012-13': 'Miami Heat', '2013-14': 'San Antonio Spurs',
    '2014-15': 'Golden State Warriors', '2015-16': 'Cleveland Cavaliers',
    '2016-17': 'Golden State Warriors', '2017-18': 'Golden State Warriors',
    '2018-19': 'Toronto Raptors', '2019-20': 'Los Angeles Lakers',
    '2020-21': 'Milwaukee Bucks', '2021-22': 'Golden State Warriors',
    '2022-23': 'Denver Nuggets', '2023-24': 'Boston Celtics',
    '2024-25': '?'
}
data['CHAMPION'] = data.apply(
    lambda row: 1 if champions.get(row['SEASON']) == row['TEAM_NAME_base'] else 0,
    axis=1
)

# === 5. Dataset split (2005–2020 for training) ===
train_data = data[(data['SEASON'] >= '2000-01') & (data['SEASON'] <= '2018-19')].copy()
test_data = data[data['SEASON'] > '2018-19'].copy()

numeric_cols = data.columns.difference(['TEAM_ID', 'TEAM_NAME_base', 'SEASON', 'CHAMPION'])
feature_cols = numeric_cols.tolist()

# No normalization, keep original feature values
# scaler = StandardScaler()
# train_data[feature_cols] = scaler.fit_transform(train_data[feature_cols])
# test_data[feature_cols] = scaler.transform(test_data[feature_cols])

X_train = train_data[feature_cols]
X_test = test_data[feature_cols]
y_train = train_data['CHAMPION']
group_train = train_data.groupby('SEASON').size().to_list()


# === 6. Utilize XGBoost Ranker model ===
# model = LGBMRanker(
#     objective='lambdarank',
#     metric='ndcg',
#     learning_rate=0.05,
#     num_leaves=31,
#     max_depth=-1,
#     n_estimators=1000,
#     min_child_samples=20,
#     lambda_l1=0.1,
#     lambda_l2=0.1,
#     random_state=42
# )

# model.fit(
#     X_train,
#     y_train,
#     group=group_train
# )

model = XGBRanker(
    objective='rank:ndcg',  # Use 'rank:ndcg' for ranking tasks
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=6,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    group=group_train,
    eval_set=[(X_train, y_train)],
    eval_group=[group_train],
    verbose=False
)

# === 7. Results prediction ===
test_data['PREDICTED_SCORE'] = model.predict(X_test)
    
# === 8. List Top-3 evaluation ===
print("\n=== Top-3 teams per season ===")
for season, group in test_data.groupby('SEASON'):
    print(f"\nSeason：{season}")
    top3 = group.sort_values('PREDICTED_SCORE', ascending=False).head(3)
    actual = champions.get(season, '?')
    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        team = row['TEAM_NAME_base']
        score = row['PREDICTED_SCORE']
        tag = "(Actual Champion)" if team == actual else ""
        print(f"Top {rank}: {team} {tag}")
    print(f"Actual champion：{actual}")


# === 9. Top-5 accuracy evaluation ===
print("\n=== Top-5 accuracy ===")
hit_count = 0
seasons_checked = 0

for season, group in test_data.groupby('SEASON'):
    actual = champions.get(season)
    if actual == '?':
        continue
    top5 = group.sort_values('PREDICTED_SCORE', ascending=False).head(5)
    if actual in top5['TEAM_NAME_base'].values:
        hit_count += 1
    seasons_checked += 1

print(f"Top-5 accuracy：{hit_count}/{seasons_checked} = {hit_count/seasons_checked:.2%}")