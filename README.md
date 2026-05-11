# NFL Expert Bias Mitigation  

A machine learning–powered tool for generating, comparing, and evaluating fantasy football player rankings.  
This project combines player historical stats, ESPN projections, and custom ML models to identify undervalued and overvalued players for the upcoming season.

This project is the functional implementation of my research paper, "Beyond the Expert: An Algorithmic Approach to Correcting Expert Bias in Fantasy Football Projections," which is published in the [Wharton Sports Analytics Journal] (https://wsb.wharton.upenn.edu/wharton-sports-analytics-journal/2026-spring-edition/).

---

## Research

### `Vishnu_Jayanti_Fantasy_Football_Research_Paper.pdf`
- **Objective**: Investigates whether "pure" algorithmic models can mitigate cognitive biases inherent in "hybrid" expert-adjusted projections like those from ESPN.
- **Methodology**: Develops four position-specific **XGBoost regression models** trained on NFL data spanning from 2013 to 2023.
- **Feature Engineering**: Utilizes over **1,900 unique variables**, including custom "career-max" indicators and lagged temporal inputs to identify player talent ceilings and performance trajectories.
- **Evaluation**: Benchmarks model performance against ESPN's preseason projections using **Mean Absolute Error (MAE)** for ranking distance and **Spearman’s Rank Correlation Coefficient (SRCC)** for ordinal ranking integrity.
- **Key Findings**:
    - The "pure" ML approach outperformed industry experts in projecting **Quarterbacks (QB)** and **Tight Ends (TE)**.
    - Algorithmic models proved less susceptible to "narrative-driven noise" and media hype that often skew human-adjusted rankings.
    - Demonstrates that machine learning can serve as a cost-effective, objective supplement to traditional expert analysis in sports forecasting.

---

## How It Works  

The project is composed of five main scripts:  

### `rankingGenerator.py`  
- Trains **XGBoost regression models** (wrapped in `MultiOutputRegressor`) on past player stats.  
- Generates predictions for **QB, RB, WR, TE** for the 2024 season.  
- Outputs position-specific predictions (`qb_predictions.csv`, `rb_predictions.csv`, etc.) ranked by fantasy points.  

### `adpGenerator.py`  
- Uses the `espn_api` package to fetch ESPN’s projected points for free agents.  
- Exports ESPN-based positional rankings (`espn_qb_predictions.csv`, `espn_rb_predictions.csv`, etc.).

### `espnFinal.py`  
- Uses the `espn_api` package to fetch fantasy football total points for free agents.  
- Exports ESPN-based positional rankings (`espn_qb_final.csv`, `espn_rb_final.csv`, etc.).

### `playerValuer.py`  
- Compares the **model-generated rankings** with **ESPN’s rankings**.  
- Identifies **undervalued** and **overvalued** players based on rank differences.  
- Outputs results into a single Excel file (`ranking_comparison.xlsx`) with separate sheets per position.

### `resultsViewer.py`  
- Compares both the **model-generated rankings** and **ESPN’s rankings** with the **final rankings**.  
- Uses **Mean Absolute Error** and **Spearman's Rank Correlation** to analyze the rankings.  
- Prints results along with the top 5 misses of both ESPN and the XGBoost model.  

### `outlierAdjustedResultsViewer.py`  
- Compares both the **model-generated rankings** and **ESPN’s rankings** with the **final rankings**.  
- Uses **Mean Absolute Error** and **Spearman's Rank Correlation** to analyze the rankings.  
- Removes 7 key NFL players who missed the entire season due to injury.  

### `modelPlot.py`  
- Creates numerous plots to visually interpret data

---

## Outputs  

- `rankings/*_predictions.csv` → Model-based predictions for each position.  
- `rankings/espn_*_predictions.csv` → ESPN projections for each position.
- `rankings/espn_*_final.csv` → Final rankings for each position.  
- `rankings/ranking_comparison.xlsx` → Combined comparison, split into undervalued/overvalued players.  
- `graphs/*.png` → Graphs to visually interpret models
- `models/*.json` → Model stored as JSON object

---

## Tech Stack  

- **Python**  
- **Pandas** – data wrangling  
- **XGBoost** – regression modeling  
- **scikit-learn** – training & evaluation utilities  
- **espn_api** – fetch ESPN projections
- **SciPy** - calculate using Spearman's formula
- **MatplotLib** - create visual data

---

## Features  

- Machine learning–based player stat prediction
- ESPN projections scraping & integration
- Identification of undervalued and overvalued players 
- Export to Excel for easy draft prep

