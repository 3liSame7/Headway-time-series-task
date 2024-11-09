import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

models = {}

model_save_path = 'E:\\Headway\\Time series task\\saved models'

def load_models(model_dir):
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
        dataset_id = data.get('dataset_id')
        values = data.get('values')

        # Directly check if dataset_id is valid without modifying it
        if dataset_id not in models:
            return jsonify({'error': f'Model for dataset {dataset_id} not found. Please provide a valid dataset ID.'}), 404

        # Prepare the input data
        test_df = pd.DataFrame(values)
        if 'value' not in test_df.columns:
            return jsonify({'error': 'Invalid input format. "values" must contain a "value" column.'}), 400

        test_df['lag_1'] = test_df['value'].shift(1)
        test_df['lag_2'] = test_df['value'].shift(2)
        test_df['rolling_mean_3'] = test_df['value'].rolling(window=3).mean()
        test_df['diff'] = test_df['value'].diff()
        test_df.dropna(inplace=True)

        # Ensure there's enough data for predictions after feature engineering
        if test_df.shape[0] == 0:
            return jsonify({'error': 'Not enough data after feature engineering.'}), 400

        X_test = test_df[['lag_1', 'lag_2', 'rolling_mean_3', 'diff']]

        # Make predictions
        model = models[dataset_id]
        predictions = model.predict(X_test)

        # Return the last prediction
        last_prediction = predictions[-1] if len(predictions) > 0 else None
        return jsonify({'prediction': last_prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
