import pandas as pd
import numpy as np
import joblib

# Load test data and test IDs
test = pd.read_csv("C:/Users/alpya/Documents/house_price_prediction_project/data/test_clean.csv")
test_ids = pd.read_csv("C:/Users/alpya/Documents/house_price_prediction_project/data/test.csv")["Id"]  # Adjust if your ID column is named differently

# Load the trained model
model = joblib.load("final_xgb_model.pkl")

# Predict log prices for the test set
log_preds = model.predict(test)

# Convert log predictions back to original scale
final_preds = np.expm1(log_preds)

# Prepare submission DataFrame
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": final_preds
})

# Save submission file
submission.to_csv("submission.csv", index=False)
print("✅ Submission saved as submission.csv")
