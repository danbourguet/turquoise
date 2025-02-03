#!/bin/bash

# --- Test for /predict_rate ---
# input data
input_data="provider_state=FL&cpt=92929&rev_code=360&payer_name=Aetna&total_beds=100"
url="http://127.0.0.1:5000/predict_rate?$input_data"

response=$(curl "$url")

predicted_rate=$(echo "$response" | grep '"predicted_rate"' | cut -d ':' -f 2 | tr -d '}')
echo "Predicted Rate: $predicted_rate"

# --- Test for /average_rate ---
input_data="provider_id=001&payer_name=Aetna"
url="http://127.0.0.1:5000/average_rate?$input_data"

response=$(curl "$url")

average_rate=$(echo "$response" | grep '"average_rate"' | cut -d ':' -f 2 | tr -d '}')
echo "Average Rate: $average_rate"

# --- Test for Total Rates by State ---
url="http://127.0.0.1:5000/rates_by_state"

response=$(curl "$url")

echo "$response"



