import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. Load Data ---
# Load the dataset from the provided CSV file
try:
    df = pd.read_csv('house_price_regression_dataset.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'house_price_regression_dataset.csv' not found. Please ensure the file is in the correct directory.")
    exit() # Exit if the file is not found

# --- 2. Data Exploration and Cleaning ---
print("\n--- Initial Data Exploration ---")
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

# Check for any missing values. If there are, we'll fill numerical ones with the mean.
# For this specific dataset, based on the snippet, it's likely clean, but this is good practice.
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        if df[column].isnull().any():
            df[column].fillna(df[column].mean(), inplace=True)
            print(f"Filled missing values in '{column}' with its mean.")
    elif df[column].dtype == 'object':
        if df[column].isnull().any():
            df[column].fillna(df[column].mode()[0], inplace=True)
            print(f"Filled missing values in '{column}' with its mode.")

print("\nMissing values after handling:")
print(df.isnull().sum())

# --- 3. Feature Engineering (and Selection) ---
# For linear regression, we'll use all numerical columns as features,
# except for the target variable 'House_Price'.
# Identify features (X) and target (y)
X = df.drop('House_Price', axis=1)
y = df['House_Price']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# --- 4. Model Training ---
print("\n--- Model Training ---")
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully.")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

# --- 5. Model Evaluation ---
print("\n--- Model Evaluation ---")
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# --- 6. Visualization ---
print("\n--- Visualizing Results ---")

# Plotting Actual vs. Predicted Prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Diagonal line
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs. Predicted House Prices")
plt.grid(True)
plt.show()

# Plotting Residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted House Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted House Prices")
plt.grid(True)
plt.show()

print("\n--- Project Completed ---")
print("The linear regression model has been trained and evaluated.")
print("The visualizations show the relationship between actual and predicted prices, and the distribution of residuals.")
