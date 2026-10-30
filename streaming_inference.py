
import pandas as pd
import joblib
import time

# 1. Load saved model
model = joblib.load("model.pkl")

# 2. Load unseen passenger records
records = pd.read_csv("unseen_records.csv")

# 3. Select 10 unseen passengers
stream_records = records.head(10)

print("====================================")
print("       STREAMING INFERENCE")
print("====================================")

# 4. Process records one by one
for index, row in stream_records.iterrows():

    passenger_id = row["PassengerId"]

    # Prepare one passenger record
    passenger = pd.DataFrame([
        row.drop("PassengerId")
    ])

    # Generate prediction
    prediction = model.predict(passenger)[0]

    result = (
        "Survived"
        if prediction == 1
        else "Not Survived"
    )

    # Display prediction
    print(
        f"Incoming Passenger ID: {passenger_id} | "
        f"Prediction: {result}"
    )

    # Wait one second
    time.sleep(1)

print("\n====================================")
print("STREAMING INFERENCE COMPLETED")
print("Total records processed:", len(stream_records))
print("====================================")
