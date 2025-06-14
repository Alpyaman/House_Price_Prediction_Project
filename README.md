# 🏡 House Price Prediction

This project is a regression analysis and machine learning pipeline designed to predict housing prices based on the Ames Housing dataset. The goal is to accurately estimate the **SalePrice** of residential homes using advanced data cleaning, feature engineering, and machine learning techniques.

---

## 📁 Project Structure
```
house_price_prediction_project/
│
├── data/
│ ├── train.csv
│ ├── test.csv
│ ├── train_clean.csv
│ └── test_clean.csv
│
├── preprocess.py # Data cleaning and advanced feature engineering
├── train_model.py # Model training and cross-validation
├── submission.py # Prediction and submission file creation
├── final_xgb_model.pkl # Saved final model (XGBoost)
├── submission.csv # Submission file for Kaggle
├── best_model_training.py # Best model training
├── best_params.py # Finding best parameters for best model
├── requirements.txt
├── log_saleprice_distribution.png # SalePrice Distribution
└──README.md
```

---

## 📊 Dataset

- Source: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)
- Train size: 1,460 entries
- Test size: 1,459 entries
- Target: `SalePrice` (log-transformed as `LogSalePrice` for modeling)

---

## 🔧 Features & Engineering

The dataset was extensively cleaned and transformed, including:

- Imputation of missing values based on feature types
- One-hot encoding for categorical features
- Log transformation of `SalePrice` for skew correction
- Creation of new features:
  - `TotalSF`: Total square footage
  - `Age`: Years since built/remodeled
  - `HasBasement`, `HasGarage`, etc.
  - Neighborhood and quality indicators

---

## 🧠 Models Trained

- Linear Regression
- Ridge Regression
- Lasso Regression
- XGBoost Regressor ✅ *(Best performing model)*

### 📈 Final Model Performance

- **Best CV RMSE**: `0.1428`  
- **Kaggle Private Score**: `0.14487`  
- **Model used**: `XGBoost Regressor` with tuned hyperparameters

---

## 🏁 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/alpyaman/house-price-prediction.git
cd house-price-prediction
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing
```bash
python preprocess.py
```

### 4. Train The Model
```bash
python train_model.py
```
### 5. Select the Best Model and Use `best_params.py` to Find Best Parameters
```bash
pyton best_params.py
```
### 6. Train the Best Model For Submission
```bash
python best_model_training.py
```
### 7. Generate Submission File
```bash
python submission.py
```
The file `submission.csv` will be saved in the root directory.

## 🧪 Future Improvements
- Implementing stacking (XGBoost + Ridge + Lasso)
- Try blending multiple model outputs
- Explore neighborhood-specific modeling
- Deploy model as a web API with Flask or FastAPI

## 📌 Requirements
- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib

## 📜 License
This project is licensed under the MIT License.

## 🙋‍♂️ Author
- *Alp Yaman*
- Data Science & ML Portfolio Project
- Feel free to connect on [Linkedln](linkedin.com/in/alp-yaman-75a901174/) or check out my other projects!
