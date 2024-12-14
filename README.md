# Uber ride fare prediction
For a more detailed explanation of the project and the code, please refer to the [Jupyter Notebook](./Uber_Fares_Prediction.ipynb) or the [PDF report](./Uber_Fares_Prediction.pdf).

## Problem description
This project focuses on analyzing Uber ride fares, including exploratory data analysis (EDA) with hypothesis testing, and building a model to predict future ride costs. The data is sourced from Kaggle: [Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset/data).

## Exploratory data analysis (EDA)
EDA revealed key insights, such as the strong relationship between trip distance and fare, and guided the removal of anomalies (e.g., negative or excessively high fares).

![Fare Distribution](exports/EDA/Distribution%20of%20fare%20amount.png)

## Data processing
Data preparation steps included:
- Removing outliers and ensuring geographic coordinates were realistic.
- Deriving new features from `pickup_datetime` (Year, Month, Day, DayOfWeek, Hour).
- Calculating `distance_km` between pickup and dropoff points.
- Computing distances to landmarks (Times Square, JFK Airport, etc.) to capture location-based effects.
- Splitting the data into training, validation, and test sets:
  - **Training set:** 117,057 samples
  - **Validation set:** 39,020 samples
  - **Test set:** 39,020 samples
- Normalizing numerical features using MinMaxScaler to ensure all values are within the same range.

## Modeling and comparison
Multiple models were tested:
- **Linear & Ridge Regression**: baseline performance (RMSE ~5.0 on validation).
- **ElasticNet**: underperformed compared to the baseline.
- **Decision Tree & Random Forest**: improved results (RMSE ~3.85+ on validation).
- **Gradient Boosting:** further improvement, with RMSE ~3.78 initially.

**Comparison table (validation set)**:
| Model             | val_rmse | val_r2  |
|-------------------|----------|---------|
| Linear Regression  | ~5.01    | ~0.72   |
| Ridge Regression   | ~5.03    | ~0.72   |
| ElasticNet         | ~9.48    | -0.00   |
| Decision Tree      | ~4.22    | ~0.80   |
| Random Forest      | ~3.85    | ~0.83   |
| Gradient Boosting  | ~3.78    | ~0.84   |

XGBoost, as the top-performing model, was chosen for final testing and hyperparameter tuning.

## Hyperparameter tuning
Systematic hyperparameter tuning involved:
- Manual adjustments and visualization-based insights for parameters like `learning_rate`, `n_estimators`, and `max_depth`.
- Automated searches using GridSearchCV and RandomizedSearchCV to find optimal configurations.

**Final parameters**:
```python
{
  'random_state': 42,
  'n_jobs': 1,
  'objective': 'reg:squarederror',
  'learning_rate': 0.05,
  'n_estimators': 100,
  'max_depth': 6,
  'subsample': 0.8,
  'colsample_bytree': 1,
  'gamma': 0,
  'reg_lambda': 1,
  'reg_alpha': 0.1
}
``` 

| Model             | val_rmse | val_r2  |
|-------------------|----------|---------|
| XGBoost Tuning     | ~3.74    | ~0.84   |
| GridSearchCV (XGB) | ~3.64    | ~0.85   |

## Test set prediction

The final XGBoost model, after tuning and validation, was tested on the unseen test dataset. The results demonstrate that the model generalizes well and maintains strong predictive performance on the test set:

- **Test RMSE**: **3.8053**  
- **Test R²**: **0.8446**  

These metrics indicate that the model effectively captures the underlying patterns in the data and is well-suited for predicting fare amounts in this use case.

## Conclusion and insights

Following a clear, step-by-step workflow ensured a systematic approach to the project, reducing errors and maintaining focus on incremental improvements at each stage. The process highlighted the importance of thorough data preprocessing and feature engineering, which provided the foundation for building an accurate predictive model.
