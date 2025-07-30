# House Price Prediction using Linear Regression
Project Overview
This project aims to build a predictive model using Linear Regression to estimate house prices based on various relevant features. It provides a hands-on experience in developing, evaluating, and interpreting a machine learning model for a regression task.

##  Features
The Python script (house-price-regression-python) in this repository performs the following key steps:

Data Collection: Loads the house_price_regression_dataset.csv into a pandas DataFrame.

Data Exploration and Cleaning:

Displays the first few rows of the dataset.

Provides a summary of the DataFrame's structure, data types, and non-null counts.

Generates descriptive statistics for numerical columns.

Identifies and handles missing values by imputing numerical columns with their mean and object (categorical) columns with their mode.

Feature Engineering & Selection: Identifies all columns (except 'House_Price') as features for the linear regression model.

## Model Training:

Splits the dataset into training and testing sets (80% training, 20% testing).

Initializes and trains a LinearRegression model from scikit-learn using the training data.

## Model Evaluation:

Makes predictions on the unseen test set.

Calculates and prints key regression evaluation metrics:

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

R-squared (R 2)
 

## Visualization: Generates insightful plots to visualize the model's performance and data characteristics:

Actual vs. Predicted House Prices: A scatter plot showing how closely predicted values align with actual values.

Distribution of Residuals: A histogram illustrating the distribution of prediction errors.

Residuals vs. Predicted House Prices: A scatter plot to check for patterns in errors and assess homoscedasticity.

## Dataset
The project uses the house_price_regression_dataset.csv file, which contains the following columns:

Square_Footage: Size of the house in square feet.

Num_Bedrooms: Number of bedrooms.

Num_Bathrooms: Number of bathrooms.

Year_Built: Year the house was built.

Lot_Size: Size of the lot.

Garage_Size: Size of the garage.

Neighborhood_Quality: A numerical indicator of the neighborhood's quality.

House_Price: The target variable, representing the price of the house.

## Results
Upon execution, the script will output:

Information about the dataset (head, info, describe).

Details on missing values and their handling.

Shapes of the training and testing datasets.

The coefficients and intercept of the trained linear regression model.

The calculated Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R 
2
 ) values.

Three visualizations demonstrating the model's performance and residual analysis.

## Key Concepts
Linear Regression: A statistical model that estimates the relationship between a dependent variable (house price) and one or more independent variables (features) by fitting a linear equation to the observed data.

Mean Squared Error (MSE): Measures the average of the squares of the errors or deviations. It's a common metric for regression models.

R-squared (R 
2
 ): Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher R 
2
  indicates a better fit.

## Residuals: The difference between the observed value and the value predicted by the model. Analyzing residuals helps in understanding the model's errors.

## Learning Objectives
This project helps in achieving the following learning objectives:

Understanding of linear regression concepts.

Practical experience in implementing a predictive model.

Model evaluation and interpretation skills.

<img width="1920" height="1080" alt="Screenshot 2025-07-30 104643" src="https://github.com/user-attachments/assets/e3ada151-6108-4c66-b952-c9387b522694" />
<img width="1920" height="1080" alt="Screenshot 2025-07-30 104654" src="https://github.com/user-attachments/assets/edccae22-125b-42dc-b639-a1a9aaaae2e2" />
<img width="1920" height="1080" alt="Screenshot 2025-07-30 104702" src="https://github.com/user-attachments/assets/1d5c9c4e-354b-4b32-be93-75ea3d02f1fe" />







