# Engage2Value: From Clicks to Conversions

Predicting purchase value from multi-session digital behavior using machine learning on real-world e-commerce analytics data.

---

## Project Overview

This project tackles a regression problem on the **Engage2Value** competition dataset — predicting `purchaseValue` for e-commerce sessions based on user behavior, device information, geographic signals, and traffic/marketing sources. The dataset contains 116,023 training sessions and 29,006 test sessions, each described by 52 raw features.

The core challenge here goes beyond modeling. The dataset is heavily sparse, has high cardinality categoricals, multi-layered missing value patterns, and an extremely right-skewed target. The bulk of this project is about making intelligent feature engineering decisions before a single model is trained.

---

## STAR Format Summary

### Situation

E-commerce businesses generate enormous amounts of clickstream data, but raw session logs are messy and full of noise — masked fields, bot-like outliers, redundant columns, and missing values that mean different things in different contexts. The business problem is straightforward: given what we know about a user's session, can we estimate how much they'll spend?

The dataset captures everything from whether the user arrived via a YouTube referral to which city they're in, their browser, and how many pages they scrolled through. Most sessions result in zero purchases, making this both a sparsity problem and a regression problem at the same time.

### Task

Build a machine learning pipeline that:
- Cleans and understands 52 raw features across four domains (behavior, geo, device, traffic)
- Engineers meaningful features from datetime fields, high-cardinality categoricals, and user history
- Prevents data leakage between train and test sets during feature construction
- Trains and evaluates multiple models, selecting the best configuration for submission

### Action

**Exploratory Data Analysis:**  
Started with a structured, domain-by-domain EDA covering user behavior signals (`totalHits`, `pageViews`, `sessionNumber`, `bounces`), geographic data (continent, country, city, metro, sub-continent), device attributes (OS, browser, device type), and traffic/marketing sources (channel, medium, referral path, AdWords data).

Key findings from EDA:
- `totals.visits` was 1 for every session — dropped as uninformative
- `pageViews` and `totalHits` had a Pearson correlation of 0.986 — retained only `pageViews`
- `totals.bounces = 1` was equivalent to `totalHits = 1` — redundant
- Sessions with a single hit never resulted in a purchase — confirmed minimum engagement threshold
- Americas drove ~33.6% conversion rate vs under 2% for most other continents
- Display channel had the highest average purchase value despite low volume
- `locationZone`, `geoNetwork.networkLocation`, and several device fields were constant across all rows — dropped
- AdWords fields had 96%+ missing values — majority dropped

**Feature Engineering (four modular functions):**

1. `user_behaviour_features`: Extracted hour, day-of-week, month, and weekend flag from `sessionStart` timestamp. Capped `sessionNumber` at 10 to reduce outlier influence. Dropped `date` after confirming 0.91 correlation with `sessionStart`.

2. `geo_features`: Grouped countries, cities, sub-continents, regions, and metros — retaining high-purchase-volume categories and collapsing the rest into "Other" or "Unknown". Lists (e.g., `top_purchase_countries`) were computed on training data only and passed as arguments to the function to prevent data leakage when applying to test data.

3. `device_features`: Grouped rare OS types (< 4500 occurrences) and browsers (< 1400 occurrences) under "Other". Dropped `device.isMobile` in favor of more granular `deviceType`. Removed constant device fields.

4. `traffic_marketing_features`: Created a `channel_group` feature (HighIntent / LowIntent / Neutral) based on observed purchase behavior per channel. Retained `trafficSource_grouped`, `referralPath_top`, `isTrueDirect`, and `isVideoAd`.

**User-level aggregation:**  
Built user history features from training data: `user_session_count`, `user_total_purchases`, `user_avg_purchase`. These features were merged into both train and test sets using `userId`. This was the most impactful feature group, as a small number of users drove disproportionate purchase value — userId 34711 alone contributed over 193 billion across 63 sessions.

**Data cleaning:**  
Removed 798 duplicate rows after all processing, reducing the dataset to 115,225 rows x 31 columns.

**Modeling:**  
Tested multiple approaches using a consistent sklearn `Pipeline` with `ColumnTransformer` (StandardScaler for numericals, OneHotEncoder for categoricals):

- Linear Regression — baseline (R² ~ 0.42)
- Ridge Regression with GridSearchCV
- Random Forest — R² ~ 0.157, underperformed
- LightGBM with GridSearchCV — best configuration
- XGBoost with RandomizedSearchCV — competitive, slightly below LightGBM
- Stacking (Ridge + LightGBM with Linear Regression meta-learner) — no meaningful gain
- Bagging and MLP were explored and discarded
- Log-transformed target (`np.log1p`) was tested
- PCA (70 components) was tested but did not improve over full features
- Feature selection via `SelectFromModel` was tested — full feature set outperformed it

Final submission used LightGBM trained on the full dataset with predictions clipped at 0.

### Result

- LightGBM outperformed all other models in cross-validated R² evaluation
- Feature importance confirmed that user history features (`user_avg_purchase`, `user_total_purchases`) were the dominant predictors
- `date_day_of_year` emerged as a surprisingly strong feature, indicating seasonal purchasing patterns
- Device and geographic features contributed but were secondary to user history
- The stacking ensemble did not meaningfully improve over the single LightGBM model
- Final pipeline was modular and leakage-free, with all reference lists derived strictly from training data

---

## Dataset

**Source:** Engage2Value — From Clicks to Conversions (Kaggle competition)

| File | Description |
|------|-------------|
| `train_data.csv` | 116,023 sessions with `purchaseValue` target |
| `test_data.csv` | 29,006 sessions without target |
| `sample_submission.csv` | Submission format |

- 52 raw columns across behavior, geo, device, and traffic domains
- Target: `purchaseValue` (highly right-skewed, majority zero)
- Roughly 93% of sessions had a purchase value of zero

---

## Feature Engineering Summary

| Domain | Key Engineered Features |
|--------|------------------------|
| User Behavior | sessionHour, sessionDayOfWeek, sessionMonth, isWeekend, sessionNumber_capped, date_day_of_year, date_year |
| Geography | country_grouped, city_grouped, subContinent_grouped, region_grouped, metro_grouped, continent_grouped |
| Device | os_grouped, browser_grouped (rare values collapsed to "Other") |
| Traffic | channel_group (HighIntent/LowIntent/Neutral), trafficSource_grouped, referralPath_top, isTrueDirect, isVideoAd |
| User History | user_session_count, user_total_purchases, user_avg_purchase |

---

## Models Evaluated

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline, R² ~ 0.42 |
| Ridge Regression | Regularized linear baseline |
| Random Forest | R² ~ 0.157, underperformed |
| LightGBM | Best performer, used as final model |
| XGBoost | Competitive, slightly below LightGBM |
| Stacking (Ridge + LightGBM) | No meaningful gain over LightGBM alone |
| Bagging, MLP | Discarded |
| LightGBM + PCA | Worse than full feature set |
| LightGBM + SelectFromModel | Worse than full feature set |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn, LightGBM, XGBoost |
| Model Selection | GridSearchCV, RandomizedSearchCV |
| Preprocessing | ColumnTransformer, StandardScaler, OneHotEncoder, Pipeline |
| Environment | Kaggle Notebooks |

---

## Project Structure

```
engage2value/
│
├── 22f2000697-notebook-t22025.ipynb    # Main notebook
├── sample_submission.csv               # Submission format
└── README.md
```

---

## Key Design Decisions

**Preventing data leakage in geo features:**  
The grouping lists (which countries to retain, which sub-continents to collapse) were computed on training data first, then passed as function arguments when transforming the test set. Computing them inside the function would have caused leakage by letting test-set distribution influence the encoding.

**Why LightGBM over others:**  
Tree-based gradient boosting handles skewed targets, mixed feature types, and sparse data better than linear models or distance-based methods. LightGBM specifically handles high-cardinality categoricals efficiently after encoding and scales well on this dataset size.

**User history features as the strongest signal:**  
Aggregated user behavior (average past purchase, total purchases, session count) was derived from training data and merged on `userId`. This is the most direct proxy for purchase intent — users who have bought before are far more likely to buy again at similar price points.

**Why full features beat PCA and SelectFromModel:**  
After preprocessing, the feature set was around 30 columns — manageable. PCA loses interpretability and can damage the feature interactions that tree models rely on. SelectFromModel based on LGBM importance excluded features that contributed indirectly through combinations, not in isolation.

---

## Notable EDA Observations

- Sessions with only 1 total hit had zero purchases — minimum engagement is a hard threshold for conversion
- Americas had a 33.6% session-to-purchase conversion rate; all other continents were under 2%
- Display channel had the highest average purchase value per session on desktop (~219M), despite being one of the lowest-traffic sources
- A small number of users drove outsized revenue — the top user alone accounted for 193 billion in total purchases
- `date_day_of_year` was among the top LightGBM feature importances, confirming strong seasonal patterns

---

## Limitations and Next Steps

- The target is extremely skewed — a two-stage model (first classify whether a purchase happens, then regress on the value) is the natural improvement but was not implemented here
- User history features are only available for users already seen in training; new users fall back to zero aggregates
- Several AdWords and campaign fields were dropped due to high missingness; better imputation strategies might recover some signal
- No explicit handling of outlier users who appear hundreds of times; capping or separate treatment could reduce noise in model training

---

## Author

**Aryan**  
Aspiring Data Scientist | Building end-to-end ML pipelines on real-world competition data

---

## License

Dataset sourced from the Engage2Value Kaggle competition. Used for educational and portfolio purposes.
