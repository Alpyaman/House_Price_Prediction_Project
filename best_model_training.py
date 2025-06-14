import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib  # for saving the model if you want

# Load full cleaned training data
df = pd.read_csv("C:/Users/alpya/Documents/house_price_prediction_project/data/train_clean.csv")
X_train = df.drop(columns=["LogSalePrice"])
y_train = df["LogSalePrice"]

# Define best model pipeline with tuned parameters
best_model = make_pipeline(
    StandardScaler(),
    XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42
    )
)

# Train the model on full training data
best_model.fit(X_train, y_train)

# (Optional) Save the model pipeline for future use
joblib.dump(best_model, "final_xgb_model.pkl")
print("✅ Final model trained and saved as 'final_xgb_model.pkl'")
