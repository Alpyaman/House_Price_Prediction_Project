import pandas  as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# Load the cleaned datasets
df = pd.read_csv('C:/Users/alpya/Documents/house_price_prediction_project/data/train_clean.csv')

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Missing values in the dataset:")
print(missing)
# Fill missing BsmtQual with 0 (meaning no basement)
df['BsmtQual'] = df['BsmtQual'].fillna(0)


# Separate features and target variable
X = df.drop(columns=['LogSalePrice'])
y = df['LogSalePrice']

# Define models to evaluate
models = {
    'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
    'Ridge Regression': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    'Lasso Regression': make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
    'XGBoost Regressor': XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
}

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse.mean()

# Fit and evaluate each model
results = {}
for name, model in models.items():
    score = rmse_cv(model)
    results[name] = score
    print(f"{name}: RMSE = {score:.4f}")

# Find the best model
best_model_name = min(results, key=results.get)
best_model_score = results[best_model_name]
print(f"\nBest Model: {best_model_name} with RMSE = {best_model_score:.4f}")