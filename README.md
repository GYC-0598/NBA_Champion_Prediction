# NBA Championship Contender Ranking with XGBoost

A machine learning project that ranks NBA teams by their likelihood of being championship contenders using **regular-season team statistics** and a **learning-to-rank model**.

Rather than treating each team as an independent binary classification problem, this project uses **XGBoost's `XGBRanker`** to rank teams *within the same NBA season*. The output is a ranked shortlist of contenders, evaluated by whether the eventual NBA champion appears in the model's **top five**.

> **Scope:** Because the features come from completed regular seasons, this is a **post-regular-season contender ranking** project, not a preseason prediction system. Ranking scores are relative model scores, **not calibrated championship probabilities**.

## Project Overview

- **Data source:** NBA Stats API, accessed through [`nba_api`](https://github.com/swar/nba_api)
- **Seasons retrieved:** 2000–01 through 2024–25 (25 seasons)
- **Training period:** 2000–01 through 2018–19
- **Test period:** 2019–20 through 2024–25
- **Model:** XGBoost `XGBRanker` with the `rank:ndcg` objective
- **Evaluation:** Top-5 hit rate, measured across seasons with a specified champion label

## Methodology

### 1. Collect team statistics

For each season, the script requests NBA regular-season team data from two NBA Stats API views:

- **Base statistics:** field goal attempts and percentages, three-point shooting, free throws, rebounds, assists, turnovers, steals, blocks, fouls, points, and win percentage.
- **Advanced statistics:** offensive and defensive ratings, net rating, effective field goal percentage, true shooting percentage, rebounding percentages, Player Impact Estimate (PIE), and assist-related metrics.

The two datasets are joined using `TEAM_ID`, and each row is tagged with its season.

### 2. Engineer features

Alongside the selected base and advanced metrics, the script calculates:

| Feature | Definition |
| --- | --- |
| `PTS_per_FGA` | Points per game / field goal attempts per game |
| `AST_TOV_ratio` | Assists per game / (turnovers per game + 1e-5) |

Each team-season receives a binary `CHAMPION` label: **1** for the NBA champion and **0** for every other team.

### 3. Train a learning-to-rank model

The dataset is divided chronologically to evaluate the model on later, unseen seasons. Each NBA season forms a separate **ranking group**, allowing `XGBRanker` to learn which teams should rank above others **within that season**.

Key hyperparameters:

| Parameter | Value |
| --- | --- |
| Objective | `rank:ndcg` |
| Learning rate | `0.05` |
| Number of estimators | `1000` |
| Maximum tree depth | `6` |
| Subsample | `1.0` |
| Column sample by tree | `1.0` |
| Random state | `42` |

No feature normalization is applied. The script also includes a **commented-out LightGBM `LGBMRanker` alternative** for experimentation.

### 4. Rank and evaluate teams

For each test season, the script:

1. Generates a ranking score for every team.
2. Prints the **top three** teams, marking the actual NBA champion when present.
3. Computes **Top-5 accuracy** across labeled test seasons:

```text
Top-5 accuracy = (number of seasons where the champion ranks in the top 5)
                 / (number of evaluated seasons)
```

**Important:** The champion entry for `2024-25` is currently set to `?` in the provided script, so that season is **excluded from Top-5 accuracy** until its label is updated. It is still included in the ranking output. No accuracy figure is reported here because actual execution results were not provided.

## Getting Started

### Requirements

- Python 3.10+ recommended
- Internet access to retrieve statistics from the NBA Stats API

Install dependencies:

```bash
pip install pandas numpy nba_api xgboost lightgbm
```

`lightgbm` is required by the current import statement, even though the LightGBM model code is commented out. You can omit it from the installation if you also remove the `LGBMRanker` import.

### Run

Save the Python script as `nba_champion_ranker.py` (or substitute its actual filename), then run:

```bash
python nba_champion_ranker.py
```

The script fetches the season data, trains the ranker, prints the top-three rankings for each test season, and reports the Top-5 hit rate. Fetching 25 seasons involves multiple API requests and may take time or encounter rate limits.

## Limitations

- **Not a preseason forecast:** Full regular-season statistics are already known at prediction time.
- **Small number of positive examples:** Only one team is labeled champion per season, limiting the number of positive training cases.
- **Ranking is not probability estimation:** `PREDICTED_SCORE` is a relative ranking score, not a championship win probability.
- **Historical changes:** Team identities, playing styles, and league conditions evolve across seasons.
- **Limited evaluation:** Top-5 hit rate alone does not describe the full ranking quality; NDCG@K, MRR, and top-1/top-3 hit rates would add context.
- **Data consistency:** NBA API availability, schema changes, and team-name consistency can affect data collection or labeling.

## Potential Improvements

- Update the season-to-champion mapping and add new seasons.
- Evaluate using expanding-window, season-by-season backtesting.
- Compare XGBoost and LightGBM ranking models under the same temporal split.
- Tune hyperparameters using a validation period separate from the test seasons.
- Add playoff-relevant indicators, injury information, and other features available *before* the intended prediction date.
- Save fetched data locally to improve reproducibility and reduce repeated API calls.

## Tech Stack

**Python · pandas · NumPy · nba_api · XGBoost · LightGBM (optional alternative)**
