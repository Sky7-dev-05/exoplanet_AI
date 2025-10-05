# 🚀 Exoplanet Detection API

REST API built with Django for automatic exoplanet detection using Machine Learning.

## 👥 Team

- **Nahine**: Backend API (DRF) 
- **Powell**: Machine Learning & Data Science
- **Belange**: Django Backend & Deployment
- **Fried**: Frontend & UI/UX
- **Wéri**: Frontend Integration

---

## 📋 Description

This API allows you to:
- ✅ Predict if data corresponds to a confirmed exoplanet
- ✅ Analyze CSV files in batch
- ✅ View prediction history
- ✅ Get statistics and ML model information

---

## 🛠️ Installation

### 1. Prerequisites
```bash
Python 3.11+
pip
virtualenv (recommended)
```

### 2. Clone and Install
```bash
# Clone the project
git clone https://github.com/Sky7-dev-05/exoplanet_AI.git
cd exoplanet_api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
```bash
# Copy .env file
cp .env.example .env

# Configure database
python manage.py makemigrations
python manage.py migrate

# Create superuser (admin)
python manage.py createsuperuser
```

### 4. Launch Server
```bash
python manage.py runserver
```

Server starts on: **http://127.0.0.1:8000**

---

## 📡 API Endpoints

### Base URL: `/api/`

| Method | Endpoint | Description | Auth |
|---------|----------|-------------|------|
| POST | `/api/predict/` | Predict a planet | No |
| POST | `/api/predict-batch/` | Predict multiple planets (CSV) | No |
| GET | `/api/model-info/` | ML model information | No |
| GET | `/api/history/` | Prediction history | No |
| GET | `/api/stats/` | Global statistics | No |
| POST | `/api/retrain/` | Retrain the model | **Admin** |
| POST/GET | `/api/metrics/` | Global metrics | No |
| POST | `/api/graph1/` | Upload graph image 1 | No |
| POST | `/api/graph2/` | Upload graph image 2 | No |

---

## 🔥 Usage Examples

### 1. Predict an Exoplanet (POST /api/predict/)

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "koi_score": 0.95,
    "koi_period": 3.52,
    "koi_impact": 0.1,
    "koi_duration": 2.5,
    "koi_depth": 500,
    "koi_prad": 1.2,
    "koi_sma": 0.05,
    "koi_teq": 580,
    "koi_model_snr": 10.0
  }'
```

**Response:**
```json
{
  "prediction": "Confirmed",
  "probability": 0.92,
  "confidence": "High",
  "message": "This exoplanet is very likely confirmed (92.0% confidence)"
}
```

---

### 2. Batch Prediction (POST /api/predict-batch/)

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/predict-batch/ \
  -F "file=@exoplanets_data.csv"
```

**CSV File Example:**
```csv
koi_score,koi_period,koi_impact,koi_duration,koi_depth,koi_prad,koi_sma,koi_teq,koi_model_snr
0.95,3.52,0.1,2.5,500,1.2,0.05,580,10.0
0.88,10.3,0.3,4.2,800,2.1,0.12,620,15.2
0.45,1.5,0.8,1.0,200,0.8,0.02,550,5.5
```

**Response:**
```json
{
  "total_predictions": 3,
  "predictions": [
    {
      "prediction": "Confirmed",
      "probability": 0.92,
      "confidence": "High",
      "message": "..."
    },
    ...
  ],
  "summary": {
    "Confirmed": 2,
    "Candidate": 1,
    "False Positive": 0
  }
}
```

---

### 3. Model Information (GET /api/model-info/)

**Request:**
```bash
curl http://127.0.0.1:8000/api/model-info/
```

**Response:**
```json
{
  "version": "1.0",
  "accuracy": 0.95,
  "f1_score": 0.93,
  "trained_on": "2025-09-30T10:00:00Z",
  "features_list": [
    "orbital_period",
    "transit_duration",
    "planetary_radius",
    "star_temperature"
  ],
  "total_predictions": 1247,
  "is_active": true
}
```

---

### 4. Statistics (GET /api/stats/)

**Request:**
```bash
curl http://127.0.0.1:8000/api/stats/
```

**Response:**
```json
{
  "total_predictions": 1247,
  "confirmed": 823,
  "candidate": 312,
  "false_positive": 112,
  "confirmed_percentage": 65.99
}
```

---

## 📊 Admin Interface

Access Django admin interface:
- URL: **http://127.0.0.1:8000/admin/**
- Login with previously created superuser

Features:
- View prediction history
- Manage ML model information
- View detailed statistics

---

## 📖 Interactive API Documentation

The API includes automatic Swagger documentation:
- **Swagger UI**: http://127.0.0.1:8000/
- **ReDoc**: http://127.0.0.1:8000/redoc/

---

## 🧪 Testing

### Testing with Postman
1. Import Postman collection (to be created)
2. Test each endpoint

### Testing with cURL
See examples above

### Testing with Python
```python
import requests

url = "http://127.0.0.1:8000/api/predict/"
data = {
    "orbital_period": 3.52,
    "transit_duration": 2.5,
    "planetary_radius": 1.2,
    "star_temperature": 5800
}

response = requests.post(url, json=data)
print(response.json())
```


## 🤝 Integration with ML Module

**What Powell needs to provide:**
1. A trained model saved as `.pkl`
2. A scaler (if normalization needed) as `.pkl`
3. Complete the `predict_single()` and `predict_batch()` functions

**Once done:**
- The API will automatically use the real model
- Remove the `_simulate_prediction()` function

---
### . Deployement

1. Go to [Render.com](https://render.com)
2. Click **“New +” → “Web Service”**
3. Connect your GitHub repo
4. Set:

   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `gunicorn exoplanet_api.wsgi:application`
5. Click **Create Web Service**

After deployment, Render will give you a live URL like:
`https://exoplanet-api.onrender.com/`



## 🆘 Support

- **API Issues**: Contact Nahine
- **ML Issues**: Contact Powell
- **Deployment Issues**: Contact Belange
- **Frontend Issues**: Contact Fried/Wéri

---

## 📜 License

MIT License - NASA Challenge 2025

