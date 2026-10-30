# Titanic ML Model Deployment and FastAPI Serving

## 1. Objective

To train a Machine Learning classification model using the Titanic dataset, save the trained model as `model.pkl`, and reuse the same model for Online, Batch and Streaming Inference.

The trained model is exposed through FastAPI REST API endpoints and tested using Swagger UI and a Python client.

## 2. Dataset

**Dataset Name:** Titanic-Dataset.csv

The dataset was uploaded directly to Google Colab.

**Dataset Source:**

https://www.kaggle.com/competitions/titanic/data

**Features Used:**
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

**Target Variable:** Survived

## 3. Technologies Used

- Google Colab
- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Joblib
- FastAPI
- Uvicorn
- Pydantic
- Requests

## 4. Project Workflow

```text
       Titanic-Dataset.csv
                |
                v
        Data Preprocessing
                |
                v
          Model Training
                |
                v
             model.pkl
                |
                v
        Unseen Records
                |
       +--------+--------+
       |        |        |
       v        v        v
     Online   Batch   Streaming
   Inference Inference Inference
       |        |        |
       +--------+--------+
                |
                v
          FastAPI Server
                |
          +-----+-----+
          |           |
          v           v
       Swagger UI  Python Client
          |           |
          +-----+-----+
                |
                v
          API Responses
```

## 5. Implementation Using Google Colab

### Task 1 – Model Training

- Uploaded Titanic-Dataset.csv into Google Colab.
- Loaded and preprocessed the dataset.
- Trained a Random Forest classification model.
- Evaluated model performance.
- Saved the trained model as `model.pkl`.
- Generated `unseen_records.csv`.

### Task 2 – Online Inference

- Loaded the saved model.
- Selected one unseen passenger.
- Generated an individual prediction.

**Sample Result:**
- Passenger ID: 566
- Prediction: Passenger Not Survived

### Task 3 – Batch Inference

- Selected 20 unseen passenger records.
- Generated predictions together.
- Saved predictions into `batch_predictions.csv`.

**Total Records Processed:** 20

### Task 4 – Streaming Inference

- Selected 10 unseen passenger records.
- Processed records one by one.
- Implemented a one-second delay between predictions.

**Total Records Processed:** 10

### Task 5 – FastAPI Model Serving

Installed FastAPI, Uvicorn and Pydantic in Google Colab.

Created `app.py` and deployed the trained model through REST API endpoints.

**Available API Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/predict` | Predict passenger survival |

### Swagger UI Documentation

Open the FastAPI Swagger documentation:

**[Open FastAPI Swagger UI](http://127.0.0.1:8000/docs)**

The Swagger UI allows users to:
- Check API health.
- Submit passenger details.
- Generate survival predictions.
- Test valid and invalid requests.
- View API responses and validation errors.

### Task 6 – Python Client Testing

Created `client.py` using the Python Requests library.

Tested three scenarios:

| Test Case | HTTP Status | Result |
|---|---:|---|
| Health Check | 200 | Successful |
| Valid Prediction | 200 | Not Survived |
| Invalid Request | 422 | Validation Error |

All three client tests completed successfully.

## 6. Generated Files

| Filename | Description |
|---|---|
| `Titanic-Dataset.csv` | Original dataset |
| `model.pkl` | Trained ML model |
| `unseen_records.csv` | Unseen passenger records |
| `batch_predictions.csv` | Batch prediction results |
| `app.py` | FastAPI application |
| `client.py` | Python API client |
| `observations.md` | Assignment observations |
| `CLA4_Report.pdf` | Final assignment report |

## 7. Screenshots

The screenshots directory contains execution evidence for:

1. Model Training
2. Online Inference
3. Batch Inference
4. Streaming Inference
5. FastAPI Health Check
6. Valid Prediction
7. Invalid Request Validation
8. Python Client Testing

## 8. Conclusion

This project demonstrates an end-to-end Machine Learning deployment workflow using Google Colab.

The same trained model was successfully reused for Online, Batch and Streaming Inference. FastAPI was used for model serving, while Swagger UI and a Python client were used to test API functionality and input validation.



https://github.com/kasim3258/CLA4-ML-Model-Deployment
