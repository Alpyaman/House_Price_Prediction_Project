import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load clean training data
df = pd.read_csv("C:/Users/alpya/Documents/house_price_prediction_project/data/train_clean.csv")
X = df.drop(columns=["LogSalePrice"])
y = df["LogSalePrice"]

# Build pipeline
pipeline = make_pipeline(
    StandardScaler(),
    XGBRegressor(objective='reg:squarederror', random_state=42)
)

# Define hyperparameter grid
param_grid = {
    'xgbregressor__n_estimators': [300, 500, 700],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__max_depth': [3, 5, 7],
    'xgbregressor__subsample': [0.7, 1],
    'xgbregressor__colsample_bytree': [0.7, 1]
}

# Initialize GridSearchCV with 3-fold CV (can increase folds if you want)
grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1
)

# Run grid search
grid_search.fit(X, y)

# Best parameters and RMSE score
print("Best parameters found:", grid_search.best_params_)
print("Best CV RMSE:", np.sqrt(-grid_search.best_score_))
