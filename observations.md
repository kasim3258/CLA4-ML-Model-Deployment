
# CLA 4 – Observations

## 1. How does Online Inference differ from Batch Inference?

Online Inference processes one input record at a time and
generates an immediate prediction. Batch Inference processes
multiple records together and generates predictions in one operation.

## 2. What is the difference between Streaming Inference and Batch Inference?

Streaming Inference processes incoming records continuously,
one at a time. Batch Inference collects multiple records and
processes them together.

## 3. Why is FastAPI used for model serving?

FastAPI exposes machine learning models through REST API
endpoints. It provides automatic API documentation, fast
request processing and Pydantic data validation.

## 4. Why should the same trained model be reused for all inference strategies?

Reusing the same trained model ensures consistent predictions,
allows fair comparison between inference strategies and avoids
unnecessary retraining.
