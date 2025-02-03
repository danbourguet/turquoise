from flask import Flask, jsonify, request
import pandas as pd
import pickle

app = Flask(__name__)


# Load trained model
with open('model.pkl', 'rb') as f:
    model, encoder, data_df = pickle.load(f)

# Load xgboost model - not implemented
# with open('xgboost_model.pkl', 'rb') as f:
#     xgbr, encoder = pickle.load(f)

# Endpoint 1: Predict Rate
@app.route('/predict_rate', methods=['GET'])
def predict_rate():
    try:
        # need data validation step
        categorical_cols = ['provider_state', 'cpt', 'rev_code', 'payer_name']
        
        input_data = request.args.to_dict()
        input_df = pd.DataFrame([input_data])

        # encode the categorical columns
        encoded_data = encoder.transform(input_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

        # merge encoded data with numberical data
        input_df = pd.concat([input_df.drop(categorical_cols, axis=1), encoded_df], axis=1)

        # make the prediction
        prediction = model.predict(input_df)[0]

        return jsonify({'predicted_rate': prediction.item()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Endpoint 2: Average Rate by Provider and Plan
@app.route('/average_rate', methods=['GET'])
def average_rate():
    try:
        provider_id = request.args.get('provider_id')
        payer_name = request.args.get('payer_name')
        
        if not provider_id or not payer_name:
            return jsonify({'error': 'Missing provider_id or payer_name parameter'}), 400

        # return jsonify(data_df.head().to_dict(orient='records'))
        filtered_data = data_df[(data_df['provider_id'] == provider_id) & (data_df['payer_name'] == payer_name)]

        if filtered_data.empty:
            return jsonify({'error': 'No data found for the given provider_state and payer_name'}), 404

        # # Calculate the average rate by provider and plan
        average_rate = filtered_data['rate'].mean()

        return jsonify({'average_rate': average_rate})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Endpoint 3: Total Rates by State
@app.route('/rates_by_state', methods=['GET'])
def rates_by_state():
    try:
        # Group by provider_state and sum the rates
        state_rates = data_df.groupby('provider_state')['rate'].sum().reset_index()

        # Convert to dictionary for JSON response
        state_rates_dict = state_rates.to_dict(orient='records')

        return jsonify(state_rates_dict)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)