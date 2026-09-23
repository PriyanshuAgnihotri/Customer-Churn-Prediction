# Customer Churn Prediction

Production-oriented machine learning system for predicting customer churn using behavioral, subscription, and revenue-related signals.

The project combines a modular ML pipeline with a FastAPI prediction service, automated testing, Docker containerization, GitHub Actions CI/CD, Docker Hub publishing, performance benchmarking, and a publicly deployed API.

---

## 🚀 Project Overview

Customer churn is a critical business problem for subscription-based organizations. This project demonstrates an end-to-end machine learning system that predicts the probability of customer churn and converts the prediction into an actionable risk classification.

The system covers the complete workflow:

**Data Generation → Feature Engineering → Model Training → Model Prediction → FastAPI → Validation → Docker → CI/CD → Docker Hub → Public Deployment**

### Key capabilities

* Synthetic B2B SaaS customer dataset generation
* Modular feature engineering pipeline
* Random Forest churn classification model
* Churn probability prediction
* Automated risk classification
* FastAPI REST API
* Pydantic request/response validation
* Automated pytest testing
* Docker containerization
* GitHub Actions CI/CD
* Docker Hub image publishing
* Interactive Swagger/OpenAPI documentation
* Local Docker end-to-end verification
* Measured API performance benchmarking
* Public HTTPS deployment on Render

---

## 🌐 Live Demo

### Public API

```text
https://customer-churn-api-qmev.onrender.com
```

### Interactive Swagger Documentation

```text
https://customer-churn-api-qmev.onrender.com/docs
```

### Public endpoints

```text
GET  /health
POST /predict
```

The deployed service was publicly verified through the live `/health`, `/docs`, and `/predict` endpoints.

---

## 🏗️ Architecture

The system uses a deliberately simple architecture so each stage can be tested and measured independently.

```text
                    ┌─────────────────────┐
                    │ Synthetic Customer  │
                    │       Data          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Random Forest Model │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Trained Model       │
                    │ random_forest_model │
                    │       .pkl          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FastAPI API     │
                    │ /health   /predict  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Docker        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    GitHub Actions   │
                    │ Tests → Build → Push│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Docker Hub      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Render Deployment  │
                    │     HTTPS API       │
                    └─────────────────────┘
```

A visual architecture diagram is maintained at:

```text
docs/architecture.png
```

---

## 📊 Dataset

The project currently uses a synthetic B2B SaaS customer dataset containing:

* **10,000 customer records**
* **17 original source dataset features**

The original source features cover customer identity, dates, subscription, revenue, and behavioral signals.

### Original source features

* Customer ID
* Signup date
* Snapshot date
* Tenure
* Country
* Industry
* Company size
* Subscription plan
* Contract type
* Monthly revenue
* Auto-renewal status
* Monthly logins
* Active days
* Average session duration
* Feature usage score
* Days since last login
* Usage change over 30 days

### API prediction inputs

The prediction API accepts **19 raw customer input fields**. These include the source customer attributes plus support, resolution, satisfaction, payment-failure, and payment-method fields required by the prediction service.

Feature engineering then creates **8 derived features** used by the model.

This distinction is intentional:

```text
17 original source dataset features
        +
API-specific prediction inputs
        ↓
19 raw API inputs
        ↓
8 derived features
        ↓
Model-ready feature set
```

---

## 🤖 Machine Learning

The current model is a **Random Forest classifier**.

### Model configuration

```text
n_estimators       = 300
max_depth          = 10
min_samples_leaf   = 5
class_weight       = balanced
random_state       = 42
n_jobs              = -1
```

The training pipeline uses:

* Numeric feature passthrough
* OneHotEncoder for categorical features
* `handle_unknown="ignore"`
* Stratified 80/20 train/test split
* A scikit-learn preprocessing/model pipeline

The trained pipeline is stored as:

```text
api/random_forest_model.pkl
```

### Prediction workflow

```text
Raw Customer Data
        ↓
Pydantic Validation
        ↓
Feature Engineering
        ↓
Preprocessing
        ↓
Random Forest
        ↓
Churn Probability
        ↓
Risk Classification
        ↓
API Response
```

---

## ⚡ FastAPI

The prediction service is implemented using FastAPI.

### Health Check

```http
GET /health
```

Example:

```json
{
  "status": "healthy"
}
```

### Churn Prediction

```http
POST /predict
```

The endpoint accepts validated customer information and returns:

* Churn probability
* Risk classification

Example response:

```json
{
  "churn_probability": 0.2571,
  "risk_level": "Medium"
}
```

The value above is an example response captured from the **publicly deployed API during release verification**. It is not a model-accuracy metric.

### API Documentation

Local Swagger:

```text
http://localhost:8000/docs
```

Public Swagger:

```text
https://customer-churn-api-qmev.onrender.com/docs
```

OpenAPI specification:

```text
http://localhost:8000/openapi.json
```

---

## 🧪 Testing

The project uses **pytest** for automated API and validation testing.

Current automated test coverage includes:

* Health endpoint
* Valid prediction request
* Negative tenure validation
* Invalid revenue validation
* Invalid feature score validation
* Invalid active days validation
* Invalid satisfaction score validation
* Missing required fields

Current test status:

```text
8 tests passing
```

Testing is also executed through GitHub Actions.

---

## 📈 Performance Benchmark

Performance was measured against the Dockerized API locally using ApacheBench (`ab`) with **1,000 requests** per test.

### `/health`

| Concurrency | Requests | Failed | Throughput | Mean | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1,000 | 0 | 1,156.84 req/s | 8.64 ms | 6 ms | 14 ms | 145 ms |

### `/predict`

| Concurrency | Requests | Failed | Throughput | Mean | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1,000 | 0 | 35.56 req/s | 281.21 ms | 266 ms | 384 ms | 657 ms |
| 25 | 1,000 | 0 | 34.04 req/s | 734.37 ms | 707 ms | 1,051 ms | 1,345 ms |
| 50 | 1,000 | 0 | 34.42 req/s | 1,452.75 ms | 1,453 ms | 1,872 ms | 1,994 ms |

### Benchmark interpretation

* `/predict` sustained approximately **34–36 requests/second** across concurrency levels of 10–50 in this local test.
* All 3,000 `/predict` benchmark requests completed successfully with **0 failed requests**.
* Increasing concurrency increased latency substantially without materially increasing throughput.
* The benchmark was performed against a local Docker container on a development machine.
* These measurements are **not production capacity or SLA guarantees**.

---

## 🐳 Docker

The API is containerized using Docker to provide a reproducible runtime environment.

### Build image

```bash
docker build -t customer-churn-prediction .
```

### Run container

```bash
docker run --rm -p 8000:8000 customer-churn-prediction
```

### Verify container

```bash
curl http://localhost:8000/health
```

Expected:

```json
{
  "status": "healthy"
}
```

Swagger:

```text
http://localhost:8000/docs
```

### Published image

```text
priyanshuagnihotri/customer-churn-api:latest
```

Pull:

```bash
docker pull priyanshuagnihotri/customer-churn-api:latest
```

Run:

```bash
docker run --rm -p 8000:8000 priyanshuagnihotri/customer-churn-api:latest
```

---

## 🔄 CI/CD

GitHub Actions automates the core software delivery workflow.

```text
Git Push
   ↓
GitHub Actions
   ↓
Run Tests
   ↓
Build Docker Image
   ↓
Publish Docker Image
   ↓
Docker Hub
```

The CI/CD workflow validates:

* Automated tests
* Docker image build
* Docker image publishing

This provides a reproducible path from source changes to a deployable container artifact.

---

## ☁️ Public Deployment

The containerized API is deployed using **Render** from the published Docker Hub image.

Deployment flow:

```text
Docker Hub Image
      ↓
Render
      ↓
HTTPS Public Endpoint
      ↓
FastAPI
```

### Deployment configuration

```text
Service: customer-churn-api
Region: Singapore
Compute: Free
Health Check: /health
Container Port: 8000
```

### Public deployment verification

The deployed API was verified with:

```text
GET  /health  → HTTP 200
GET  /docs    → Swagger UI available
POST /predict → HTTP 200
```

A successful public prediction returned:

```json
{
  "churn_probability": 0.2571,
  "risk_level": "Medium"
}
```

Free Render instances may spin down after inactivity, so the first request after an idle period can experience a startup delay.

---

## 📦 Docker Hub

The application Docker image is published to Docker Hub:

```text
priyanshuagnihotri/customer-churn-api
```

The image can be pulled and executed independently of the local development environment.

```bash
docker pull priyanshuagnihotri/customer-churn-api:latest
```

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── api/
│   ├── app.py
│   ├── main.py
│   ├── model.pkl
│   ├── random_forest_model.pkl
│   └── schemas.py
│
├── src/
│   ├── constants/
│   ├── data/
│   ├── features/
│   ├── generators/
│   ├── models/
│   └── utils/
│
├── tests/
│   └── test_api.py
│
├── dashboard/
│   └── powerbi_dashboard.pbix
│
├── notebooks/
│   └── eda.ipynb
│
├── .github/
│   └── workflows/
│
├── docs/
│   └── architecture.png
│
├── Dockerfile
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| ML | scikit-learn |
| Model | Random Forest |
| API | FastAPI |
| Validation | Pydantic |
| Data Processing | pandas |
| Model Serialization | joblib |
| Testing | pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Image Registry | Docker Hub |
| Deployment | Render |
| API Documentation | OpenAPI / Swagger |
| Dashboard | Power BI |

---

## ▶️ Local Development

### 1. Clone repository

```bash
git clone https://github.com/PriyanshuAgnihotri/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run tests

```bash
pytest
```

Expected:

```text
8 passed
```

### 5. Start API

```bash
uvicorn api.app:app --reload
```

### 6. Open Swagger

```text
http://localhost:8000/docs
```

---

## 🔐 Production Considerations

The project is intentionally built incrementally toward production readiness.

### Current production-oriented capabilities

* Containerized application runtime
* Automated API tests
* CI/CD validation
* Pydantic request validation
* Health endpoint
* Model loaded once at API startup
* Docker image publishing
* Reproducible deployment artifact
* Measured local API performance
* Public HTTPS deployment
* Public endpoint verification

### Future improvements

Potential improvements include:

* Structured application logging
* Monitoring and observability
* Model versioning
* Model/data drift monitoring
* Authentication and authorization
* Production-grade autoscaling
* Cloud load testing
* Externalized model storage

These improvements should be introduced only when justified by measured requirements.

---

## 📈 Project Status

| Component | Status |
|---|---|
| Data Pipeline | ✅ Complete |
| Feature Engineering | ✅ Complete |
| Random Forest Model | ✅ Complete |
| FastAPI API | ✅ Complete |
| Pydantic Validation | ✅ Complete |
| Automated Tests | ✅ 8 Passing |
| GitHub Actions | ✅ Passing |
| Docker Build | ✅ Passing |
| Docker Hub Publishing | ✅ Complete |
| Docker Pull | ✅ Verified |
| Container Startup | ✅ Verified |
| Container `/predict` | ✅ E2E Verified |
| Performance Benchmarking | ✅ Complete |
| Architecture Diagram | ✅ Complete |
| Public Docker Deployment | ✅ Complete |
| Public `/health` | ✅ Verified |
| Public `/docs` | ✅ Verified |
| Public `/predict` | ✅ Verified |

---

## 🎯 Project Objective

This project is designed to demonstrate practical machine learning engineering rather than only model training.

The goal is to build a reproducible system that connects:

**Machine Learning + Software Engineering + API Development + Testing + Containerization + CI/CD + Public Deployment**

while maintaining a simple architecture that can be measured, tested, deployed, and extended toward production use.
