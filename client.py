
import requests

# Local FastAPI server
BASE_URL = "http://127.0.0.1:8000"

# TEST 1: Health Check
print("========== HEALTH CHECK ==========")

health = requests.get(
    BASE_URL + "/health",
    timeout=10
)

print("Status:", health.status_code)
print("Response:", health.json())


# TEST 2: Valid Prediction
print("\n========== VALID PREDICTION ==========")

passenger = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

response = requests.post(
    BASE_URL + "/predict",
    json=passenger,
    timeout=10
)

print("Status:", response.status_code)
print("Response:", response.json())


# TEST 3: Invalid Prediction
print("\n========== INVALID REQUEST ==========")

invalid_passenger = {
    "Pclass": 5,
    "Sex": "unknown",
    "Age": -10,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

invalid_response = requests.post(
    BASE_URL + "/predict",
    json=invalid_passenger,
    timeout=10
)

print("Status:", invalid_response.status_code)
print("Response:", invalid_response.json())

print("\nAPI CLIENT TESTING COMPLETED!")
