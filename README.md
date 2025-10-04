# 🚀 Exoplanet Detection API

API REST Django pour la détection automatique d'exoplanètes utilisant le Machine Learning.

## 👥 Équipe

- **Nahine** : Backend API (DRF) - C'est TOI ! 🔥
- **Powell** : Machine Learning & Data Science
- **Belange** : Backend Django & Déploiement
- **Fried** : Frontend & UI/UX
- **Wéri** : Intégration Frontend

---

## 📋 Description

Cette API permet de :
- ✅ Prédire si des données correspondent à une exoplanète confirmée
- ✅ Analyser des fichiers CSV en batch
- ✅ Consulter l'historique des prédictions
- ✅ Obtenir des statistiques et infos sur le modèle ML

---

## 🛠️ Installation

### 1. Prérequis
```bash
Python 3.11+
pip
virtualenv (recommandé)
```

### 2. Cloner et installer
```bash
# Cloner le projet
git clone <votre-repo>
cd exoplanet_api

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Configuration
```bash
# Copier le fichier .env
cp .env.example .env

# Configurer la base de données
python manage.py makemigrations
python manage.py migrate

# Créer un superuser (admin)
python manage.py createsuperuser
```

### 4. Lancer le serveur
```bash
python manage.py runserver
```

Le serveur démarre sur : **http://127.0.0.1:8000**

---

## 📡 Endpoints API

### Base URL : `/api/`

| Méthode | Endpoint | Description | Auth |
|---------|----------|-------------|------|
| POST | `/api/predict/` | Prédire une planète | Non |
| POST | `/api/predict-batch/` | Prédire plusieurs planètes (CSV) | Non |
| GET | `/api/model-info/` | Infos sur le modèle ML | Non |
| GET | `/api/history/` | Historique des prédictions | Non |
| GET | `/api/stats/` | Statistiques globales | Non |
| POST | `/api/retrain/` | Ré-entraîner le modèle | **Admin** |

---

## 🔥 Exemples d'utilisation

### 1. Prédire une exoplanète (POST /api/predict/)

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
  "message": "Cette exoplanète est très probablement confirmée (92.0% de confiance)"
}
```

---

### 2. Prédire en batch (POST /api/predict-batch/)

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/predict-batch/ \
  -F "file=@exoplanets_data.csv"
```

**Exemple de fichier CSV :**
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

### 3. Infos du modèle (GET /api/model-info/)

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

### 4. Statistiques (GET /api/stats/)

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

## 📊 Interface Admin

Accéder à l'interface admin Django :
- URL : **http://127.0.0.1:8000/admin/**
- Login avec le superuser créé précédemment

Fonctionnalités :
- Consulter l'historique des prédictions
- Gérer les infos du modèle ML
- Voir les statistiques détaillées

---

## 📖 Documentation API Interactive

L'API inclut une documentation Swagger automatique :
- **Swagger UI** : http://127.0.0.1:8000/
- **ReDoc** : http://127.0.0.1:8000/redoc/

---

## 🧪 Tests

### Tester avec Postman
1. Importer la collection Postman (à créer)
2. Tester chaque endpoint

### Tester avec cURL
Voir les exemples ci-dessus

### Tester avec Python
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

---

## 🔗 Intégration avec le Frontend

Le frontend (Fried + Wéri) doit appeler ces endpoints :

**Exemple React :**
```javascript
// Prédiction simple
const predictExoplanet = async (data) => {
  const response = await fetch('http://127.0.0.1:8000/api/predict/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return await response.json();
};

// Usage
const result = await predictExoplanet({
  orbital_period: 3.52,
  transit_duration: 2.5,
  planetary_radius: 1.2,
  star_temperature: 5800
});

console.log(result); // { prediction: "Confirmed", probability: 0.92, ... }
```

---

## 🤝 Intégration avec le module ML de Powell

Le fichier `ml_model/predict_exoplanet.py` est un **template** que Powell doit compléter.

**Ce que Powell doit fournir :**
1. Un modèle entraîné sauvegardé en `.pkl`
2. Un scaler (si normalisation) en `.pkl`
3. Compléter les fonctions `predict_single()` et `predict_batch()`

**Une fois fait :**
- L'API utilisera automatiquement le vrai modèle
- Supprimer la fonction `_simulate_prediction()`

---

## 🚀 Déploiement (Belange)

### Option 1 : Docker
```dockerfile
# Dockerfile à créer
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "exoplanet_api.wsgi:application"]
```

### Option 2 : Serveur classique
```bash
# Installer gunicorn
pip install gunicorn

# Lancer en production
gunicorn exoplanet_api.wsgi:application --bind 0.0.0.0:8000
```

---

## 📝 TODO List

### Nahine (TOI)
- [x] Créer les endpoints REST
- [x] Validation des données
- [x] Gestion des erreurs
- [ ] Tests unitaires
- [ ] Documentation Postman

### Powell
- [ ] Entraîner le modèle ML
- [ ] Sauvegarder en .pkl
- [ ] Compléter `predict_exoplanet.py`
- [ ] Documenter les features attendues

### Belange
- [ ] Intégrer l'API dans le projet global
- [ ] Configuration Docker
- [ ] Déploiement serveur
- [ ] Monitoring & logs

### Fried & Wéri
- [ ] Interface web
- [ ] Intégration API REST
- [ ] Dashboard stats
- [ ] Upload CSV

---

## 🆘 Support

- **Problèmes API** : Contacter Nahine
- **Problèmes ML** : Contacter Powell
- **Problèmes Déploiement** : Contacter Belange
- **Problèmes Frontend** : Contacter Fried/Wéri

---

## 📜 Licence

MIT License - NASA Challenge 2025

---

**Bon courage Nahine ! 🚀🔥**