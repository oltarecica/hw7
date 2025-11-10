import requests
import json

API_URL = "http://127.0.0.1:8001/predict"

data_point = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

try:
    response = requests.post(API_URL, json=data_point)

    if response.status_code == 200:
        print("Prediction:", response.json())
    elif response.status_code == 404:
        print("Error: Endpoint not found (404). Check your URL.")
    elif response.status_code == 500:
        print("Error: Internal server error (500). Check your API code.")
    else:
        print(f"Unexpected response ({response.status_code}):", response.text)

except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the API. Is it running?")
except requests.exceptions.Timeout:
    print("Error: The request timed out.")
except requests.exceptions.RequestException as e:
    print("Error: An unexpected error occurred:", e)
