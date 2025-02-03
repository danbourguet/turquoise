import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


# Create synthetic data
n_samples = 1000

data = {
    'rate': np.random.randint(100, 1000, n_samples),  # Target variable
    'provider_state': np.random.choice(['FL', 'NY', 'TX', 'CA'], n_samples),
    'cpt': np.random.choice(['92929', '92928'], n_samples),
    'rev_code': np.random.choice(['360', '480', '481', '999'], n_samples),
    'payer_name': np.random.choice(['Aetna', 'Cigna', 'UnitedHealthcare'], n_samples),
    'total_beds': np.random.randint(50, 500, n_samples),
    'provider_id': np.random.choice(['001', '0240', '456'], n_samples)
}

data_df = pd.DataFrame(data)

# drop provier_id
model_df = data_df.drop(columns=['provider_id'])

# Process categorical variables
# Identify categorical columns
categorical_cols = ['provider_state', 'cpt', 'rev_code', 'payer_name']

# Create OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 

# Transform the categorical columns
encoded_data = encoder.fit_transform(model_df[categorical_cols])

# Create a new DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

# Combine the encoded features with the numerical features
model_df = pd.concat([model_df.drop(categorical_cols, axis=1), encoded_df], axis=1)

# Define X and y w test train splits
X = model_df.drop('rate', axis=1)
y = model_df['rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Sklearn Linear Regression
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
print(model.score(X_test, y_test))

### XGBoost Regressor
xgbr = xgb.XGBRegressor(objective='reg:squarederror',
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42)

xgbr.fit(X_train, y_train)

print("R-squared:", xgbr.score(X_test, y_test)) # Use the score method for R-squared

# Other evaluation methods
from sklearn.metrics import mean_squared_error, r2_score
y_pred = xgbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump((model, encoder, data_df), f)

# Save xgboost model
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump((xgbr, encoder), f)
