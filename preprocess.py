import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('C:/Users/alpya/Documents/house_price_prediction_project/data/train.csv')
test = pd.read_csv('C:/Users/alpya/Documents/house_price_prediction_project/data/test.csv')

train['source'] = 'train'
test['source'] = 'test'
test['SalePrice'] = np.nan  # Add SalePrice column to test set for consistency

df = pd.concat([train, test], axis=0)

features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
            'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'ExterQual',
            'KitchenQual', 'BsmtQual', 'Neighborhood', 'source']

df = df[features]

# Fill missing values for categorical features with 'NA'
def fill_missing_categorical(df, columns):
    for col in columns:
        df[col] = df[col].fillna('NA')

df['BsmtQual'] = df['BsmtQual'].fillna("NA")

# Map ordinal quality scores
quality_mapping = {'NA': 0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
for col in ['ExterQual', 'KitchenQual', 'BsmtQual']:
    df[col] = df[col].map(quality_mapping)

# Fill missing values for numerical features with 0
for col in ['GarageCars', 'GarageArea', 'TotalBsmtSF']:
    df[col] = df[col].fillna(0)

# Advanced Feature Engineering
df['LogSalePrice'] = np.log1p(df['SalePrice'])
df['TotalArea'] = df['GrLivArea'] + df['GarageArea'] + df['TotalBsmtSF']
df['HouseAge'] = 2025 - df['YearBuilt']
df['SinceRemod'] = 2025 - df['YearRemodAdd']

# Log transformation for skewed features
for col in ['GrLivArea', 'GarageArea', 'TotalBsmtSF', 'TotalArea']:
    df[col] = np.log1p(df[col])

# One-hot encode Neighborhood
df = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)

# Drop unneeded columns
df.drop(columns=['YearBuilt', 'YearRemodAdd', 'SalePrice'], inplace=True)

# Split back into train and test sets
train_clean = df[df['source'] == 'train'].drop(columns=['source'])
test_clean = df[df['source'] == 'test'].drop(columns=['source', 'LogSalePrice'])

# Save the cleaned datasets
train_clean.to_csv('C:/Users/alpya/Documents/house_price_prediction_project/data/train_clean.csv', index=False)
test_clean.to_csv('C:/Users/alpya/Documents/house_price_prediction_project/data/test_clean.csv', index=False)

# Plot log SalePrice distribution
plt.figure(figsize=(10, 6))
sns.histplot(train_clean['LogSalePrice'], bins=40, kde=True)
plt.title('Log SalePrice Distribution')
plt.savefig('C:/Users/alpya/Documents/house_price_prediction_project/log_saleprice_distribution.png')
plt.close()

print("Data preprocessing complete. Cleaned datasets saved.")