import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import adfuller
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

train_folder = 'E:\\Headway\\Time series task\\train_splits\\train_splits'
model_save_path = 'E:\\Headway\\Time series task\\saved models'

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

def handle_missing_data(df):
    df.fillna(method='ffill', inplace=True)  
    df.fillna(method='bfill', inplace=True)  
    return df

def feature_engineering(df):
    df = handle_missing_data(df)
    df['lag_1'] = df['value'].shift(1)
    df['lag_2'] = df['value'].shift(2)
    df['rolling_mean_3'] = df['value'].rolling(window=3).mean()
    df['diff'] = df['value'].diff()
    df.dropna(inplace=True)
    return df

def split_train_validation(train_df):
    X = train_df[['lag_1', 'lag_2', 'rolling_mean_3', 'diff']]
    y = train_df['value']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def train_and_save_model(dataset_id, X_train, y_train, X_val, y_val):
    param_grids = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {'n_estimators': [50]}
        },
        'SVR': {
            'model': SVR(),
            'params': {'kernel': ['linear'], 'C': [1]}
        },
        'XGBoost': {
            'model': XGBRegressor(objective='reg:squarederror', random_state=42),
            'params': {'n_estimators': [50], 'learning_rate': [0.1]}
        },
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {}
        }
    }

    best_rmse = float('inf')
    best_model = None

    for model_name, config in param_grids.items():
        model = config['model']
        param_grid = config['params']
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model_in_grid = grid_search.best_estimator_
        y_pred_val = best_model_in_grid.predict(X_val)
        rmse_val = mean_squared_error(y_val, y_pred_val) ** 0.5

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_model = best_model_in_grid

    model_path = os.path.join(model_save_path, f'{dataset_id}_best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Saved model for {dataset_id} with RMSE: {best_rmse}")

def process_datasets(train_folder):
    for filename in os.listdir(train_folder):
        if filename.endswith('.csv'):
            dataset_id = filename.split('.')[0]
            train_path = os.path.join(train_folder, filename)
            train_df = pd.read_csv(train_path)
            
            if 'date' in train_df.columns:
                train_df['date'] = pd.to_datetime(train_df['date'])
                train_df.set_index('date', inplace=True)
            
            train_df = feature_engineering(train_df)
            X_train, X_val, y_train, y_val = split_train_validation(train_df)
            train_and_save_model(dataset_id, X_train, y_train, X_val, y_val)

process_datasets(train_folder)
