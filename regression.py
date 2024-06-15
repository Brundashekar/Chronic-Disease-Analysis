import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def main_(df):
    
    filtered_data = df[(df['questionid'] == 'ALC1_1') & (~df['locationabbr'].isin(['US', 'VI', 'PR', 'GU']))]
    necessary_col = ['yearstart', 'yearend', 'locationabbr', 'datavalue', 'stratification1']
    filtered_data = filtered_data[necessary_col]
    filtered_data.dropna(subset=['datavalue'], inplace=True)

    X = filtered_data.drop('datavalue', axis=1)
    y = filtered_data['datavalue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    # Define the numerical transformer
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Linear Regression
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', LinearRegression())])
    lr_pipeline.fit(X_train, y_train)

    # Save the Linear Regression model
    joblib.dump(lr_pipeline, 'linear_regression_model.pkl')

    # XGB Regression
    xgb_model = XGBRegressor(random_state=42)
    xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', xgb_model)])
    xgb_pipeline.fit(X_train, y_train)

    # Save the XGB model
    joblib.dump(xgb_pipeline, 'XGB_model.pkl')

    # Random Forest Regression
    rf_model = RandomForestRegressor(random_state=42)
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', rf_model)])
    rf_pipeline.fit(X_train, y_train)

    # Save the Random Forest model
    joblib.dump(rf_pipeline, 'rf_model.pkl')

    # Make predictions and calculate metrics for Linear Regression
    lr_predictions = lr_pipeline.predict(X_test)
    lr_r2_train = r2_score(y_train, lr_pipeline.predict(X_train))
    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_rmse = np.sqrt(lr_mse)
    lr_r2 = r2_score(y_test, lr_predictions)

    # Make predictions and calculate metrics for XGB
    xgb_predictions = xgb_pipeline.predict(X_test)
    xgb_r2_train = r2_score(y_train, xgb_pipeline.predict(X_train))
    xgb_mse = mean_squared_error(y_test, xgb_predictions)
    xgb_rmse = np.sqrt(xgb_mse)
    xgb_r2 = r2_score(y_test, xgb_predictions)

    # Make predictions and calculate metrics for Random Forest
    rf_predictions = rf_pipeline.predict(X_test)
    rf_r2_train = r2_score(y_train, rf_pipeline.predict(X_train))
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(rf_mse)
    rf_r2 = r2_score(y_test, rf_predictions)

    return lr_mse, lr_r2, lr_r2_train,xgb_mse, xgb_r2, xgb_r2_train,rf_mse, rf_r2,rf_r2_train,lr_predictions, xgb_predictions, rf_predictions,X_test, y_test

def plot_model_comparisons_plotly(X_test, y_test, lr_predictions, xgb_predictions, rf_predictions):    
    lr_predictions = lr_predictions
    xgb_predictions = xgb_predictions
    rf_predictions = rf_predictions

    # Create subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Linear Regression", "XGBoost", "Random Forest"))

    # Add scatter plot for Linear Regression
    fig.add_trace(
        go.Scatter(x=y_test, y=lr_predictions, mode='markers', name='LR Predictions', marker=dict(color='blue', opacity=0.5)),
        row=1, col=1
    )
    # Add line for perfect prediction
    fig.add_trace(
        go.Scatter(x=y_test, y=y_test, mode='lines', name='Perfect Prediction', marker=dict(color='red')),
        row=1, col=1
    )

    # Add scatter plot for XGBoost
    fig.add_trace(
        go.Scatter(x=y_test, y=xgb_predictions, mode='markers', name='XGB Predictions', marker=dict(color='green', opacity=0.5)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=y_test, y=y_test, mode='lines', name='Perfect Prediction', marker=dict(color='red')),
        row=1, col=2
    )

    # Add scatter plot for Random Forest
    fig.add_trace(
        go.Scatter(x=y_test, y=rf_predictions, mode='markers', name='RF Predictions', marker=dict(color='orange', opacity=0.5)),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(x=y_test, y=y_test, mode='lines', name='Perfect Prediction', marker=dict(color='red')),
        row=1, col=3
    )

    # Update xaxis and yaxis properties
    fig.update_xaxes(title_text="Actual Values", row=1, col=1)
    fig.update_xaxes(title_text="Actual Values", row=1, col=2)
    fig.update_xaxes(title_text="Actual Values", row=1, col=3)

    fig.update_yaxes(title_text="Predicted Values", row=1, col=1)

    # Update layout
    fig.update_layout(height=500, width=1200, title_text="Actual vs Predicted Values Comparison")
    
    return fig