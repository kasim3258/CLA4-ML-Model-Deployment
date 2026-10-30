
import pandas as pd
import joblib

# 1. Load saved model
model = joblib.load("model.pkl")

# 2. Load unseen records
records = pd.read_csv("unseen_records.csv")

# 3. Select 20 passengers
batch_records = records.head(20)

# 4. Separate passenger IDs and features
passenger_ids = batch_records["PassengerId"]

X_batch = batch_records.drop(
    columns=["PassengerId"]
)

# 5. Generate batch predictions
predictions = model.predict(X_batch)

# 6. Create results dataframe
results = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Prediction": predictions
})

# 7. Convert predictions into labels
results["Result"] = results["Prediction"].map({
    0: "Not Survived",
    1: "Survived"
})

# 8. Save results
results.to_csv(
    "batch_predictions.csv",
    index=False
)

# 9. Display results
print("====================================")
print("          BATCH INFERENCE")
print("====================================")

print(results)

print("\nTotal passengers processed:", len(results))
print("File saved: batch_predictions.csv")

print("\nBatch inference completed successfully!")
