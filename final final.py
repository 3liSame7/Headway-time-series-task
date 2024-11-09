"""
Time series task
"""

import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import joblib
import warnings
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def split_train_validation(train_df):
    X = train_df[['lag_1', 'lag_2', 'rolling_mean_3', 'diff']]
    y = train_df['value']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

train_folder = 'train_splits'
test_folder = 'test_splits'
model_save_path = 'saved models'

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)



def handle_missing_data(df):
    df.fillna(method='ffill', inplace=True)  
    df.fillna(method='bfill', inplace=True)  
    return df

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def check_stationarity(df):
    result = adfuller(df['value'])
    return result[1] <= 0.05

def feature_engineering(df):
   
    df = handle_missing_data(df)

    df['lag_1'] = df['value'].shift(1)
    df['lag_2'] = df['value'].shift(2)

    df['rolling_mean_3'] = df['value'].rolling(window=3).mean()
    df['rolling_std_3'] = df['value'].rolling(window=3).std()

    df.dropna(inplace=True)


    if len(df) < 10: 
        print(f"Dataset is too small after feature engineering")
        return None

    if 'timestamp' in df.columns:
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek

    df['diff'] = df['value'].diff()


    fft_vals = np.fft.fft(df['value'].values)

    df['fft_seasonality'] = np.abs(fft_vals).mean() 
    df['fft_trend'] = fft_vals.real.mean()  

 
    if isinstance(df.index, pd.DatetimeIndex):
        df['year'] = df.index.year
        df['month'] = df.index.month  
        df['day'] = df.index.day  
        df['weekday'] = df.index.weekday  
        df['quarter'] = df.index.quarter 
        df['is_month_end'] = df.index.is_month_end.astype(int)  
        df['is_month_start'] = df.index.is_month_start.astype(int)  
        df['is_weekend'] = df.index.weekday >= 5  

    df.dropna(inplace=True)  
    
    return df

def decompose_time_series(train_dfs, max_plots=3):
    plt.figure(figsize=(15, 12)) 
    num_plots = min(len(train_dfs), max_plots)

    for i, (dataset_id, df) in enumerate(train_dfs.items()):
        if i >= num_plots:
            break

        decomposition = seasonal_decompose(df['value'], model='additive', period=12)
        plt.subplot(num_plots, 1, i + 1)  
        plt.plot(df.index, df['value'], label='Original', color='blue')
        plt.plot(decomposition.trend, label='Trend', color='orange')
        plt.plot(decomposition.seasonal, label='Seasonal', color='green')
        plt.plot(decomposition.resid, label='Residual', color='red')
        plt.title(f'Decomposition for {dataset_id}')
        plt.legend()

    plt.tight_layout()  
    plt.show() 


def train_and_save_model(dataset_id, X_train, y_train, X_val, y_val, train_df):
    np.random.seed(42)
    param_grids = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'kernel': ['linear'],  
                'C': [1]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(objective='reg:squarederror', random_state=42),
            'params': {
                'n_estimators': [50],  
                'learning_rate': [0.1]  
            }
        },
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {}  
        }
    }

    best_rmse = float('inf')
    best_model = None
    best_model_name = ""
    model_results = []

    for model_name, config in param_grids.items():
        model = config['model']
        param_grid = config['params']
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model_in_grid = grid_search.best_estimator_
        y_pred_val = best_model_in_grid.predict(X_val)
        rmse_val = mean_squared_error(y_val, y_pred_val) ** 0.5  
        model_results.append({'model': model_name, 'rmse': rmse_val, 'best_params': grid_search.best_params_})

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_model = best_model_in_grid
            best_model_name = model_name

    
    model_path = os.path.join(model_save_path, f'{dataset_id}_best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Saved {best_model_name} model for {dataset_id} with RMSE: {best_rmse} (on validation set)")

    return model_results



def process_datasets(train_folder):
    all_results = []
    train_dfs = {}  
    for filename in os.listdir(train_folder):
        if filename.endswith('.csv'):
            dataset_id = filename.split('.')[0]
            train_path = os.path.join(train_folder, filename)

            print(f"Processing {dataset_id}")

            train_df = pd.read_csv(train_path)

            
            if 'date' in train_df.columns:
                train_df['date'] = pd.to_datetime(train_df['date'])
                train_df.set_index('date', inplace=True)

            
            train_df = feature_engineering(train_df)

           
            train_dfs[dataset_id] = train_df

            
            X_train, X_val, y_train, y_val = split_train_validation(train_df)

           
            model_results = train_and_save_model(dataset_id, X_train, y_train, X_val, y_val, train_df)

            
            result_row = {'dataset_id': dataset_id}
            for result in model_results:
                result_row[result['model']] = result['rmse']

            all_results.append(result_row)

    
    decompose_time_series(train_dfs)
    return all_results


all_results = process_datasets(train_folder)

#flaskkkkkkkkkkkkkkkkkkkkkkkkkk
app = Flask(__name__)


models = {}

def load_models(model_dir):
    """Load all models from the specified directory."""
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl'):
            model_path = os.path.join(model_dir, filename)
            model_name = filename[:-4] 
            models[model_name] = joblib.load(model_path)
    print(f"Loaded models: {list(models.keys())}")


load_models(model_save_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        dataset_id = data['dataset_id']
        values = data['values']
        
        
        if dataset_id not in models:
            return jsonify({'error': f'Model for dataset {dataset_id} not found'}), 404
        
        model = models[dataset_id]
        

        test_df = pd.DataFrame(values)
        test_df['lag_1'] = test_df['value'].shift(1)
        test_df['lag_2'] = test_df['value'].shift(2)
        test_df['rolling_mean_3'] = test_df['value'].rolling(window=3).mean()
        test_df['diff'] = test_df['value'].diff()
        test_df.dropna(inplace=True)
        
        X_test = test_df[['lag_1', 'lag_2', 'rolling_mean_3', 'diff']]
        
        predictions = model.predict(X_test)
        
        last_prediction = predictions[-1] if len(predictions) > 0 else None
        
        return jsonify({'prediction': last_prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()