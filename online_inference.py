
import pandas as pd
import joblib

# 1. Load the saved trained model
model = joblib.load("model.pkl")

# 2. Load unseen passenger records
records = pd.read_csv("unseen_records.csv")

# 3. Select one unseen passenger
passenger = records.drop(
    columns=["PassengerId"]
).iloc[[0]]

passenger_id = records.iloc[0]["PassengerId"]

# 4. Generate prediction
prediction = model.predict(passenger)[0]

# 5. Display result
print("================================")
print("       ONLINE INFERENCE")
print("================================")

print("Passenger ID:", passenger_id)

print("\nPassenger Details:")
print(passenger)

print("\nPrediction Result:")

if prediction == 1:
    print("Passenger Survived")
else:
    print("Passenger Not Survived")

print("\nOnline inference completed successfully!")
