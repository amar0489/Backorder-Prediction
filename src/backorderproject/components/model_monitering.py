import requests
import time

API_URL = "http://127.0.0.1:8000/predict" 

sample_input = {
    "national_inv": 100.0,
    "lead_time": 5.0,
    "in_transit_qty": 10.0,
    "forecast_3_month": 300.0,
    "sales_1_month": 50.0,
    "min_bank": 20.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.8,
    "local_bo_qty": 5.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No"
}

for i in range(5):  # Run multiple times to measure latency
    start_time = time.time()
    response = requests.post(API_URL, json=sample_input)
    end_time = time.time()
    latency = end_time - start_time

    if response.status_code == 200:
        result = response.json()
        print(f"Test {i+1}: Latency = {latency:.4f} seconds, API Latency = {result['latency_seconds']} seconds")
    else:
        print(f"Test {i+1}: Error - {response.status_code}")
