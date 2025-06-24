
# 🚗 Fraud Detection Dashboard (Dash + FastAPI)

An interactive web application to detect fraudulent vehicle insurance claims using machine learning. The project integrates:

- 📊 **Dash** for data visualization and user interaction
- 🚀 **FastAPI** for model inference and backend
- 📦 **Docker** for easy packaging and sharing

---

## 🧰 Features

- Upload `.csv` or `.xlsx` files and predict fraudulent claims
- View prediction summary and download results as Excel
- Interactive dashboards for EDA (exploratory data analysis)
- Permutation Importance-based feature importance
- Multiple model support with model metrics display

---

## 🚀 How to Run the App

### 🧑‍💻 Option 1: Run Locally (Python)

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/fraud-detection-app.git
cd fraud-detection-app
```

2. **Create virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Start the FastAPI backend**

```bash
uvicorn main:app --reload
```

4. **Start the Dash frontend**

```bash
python app.py
```

---

### 🐳 Option 2: Run with Docker

1. **Install Docker**

- 📥 [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
- ✅ Install and launch Docker

2. **Build the Docker Image**

```bash
docker build -t fraud-detection-app .
```

3. **Run the App**

```bash
docker run -d -p 8050:8050 -p 8000:8000 fraud-detection-app
```

- Open [http://localhost:8050](http://localhost:8050) for the Dash UI
- Open [http://localhost:8000/docs](http://localhost:8000/docs) for FastAPI endpoints

4. **Stop the Container**

```bash
docker ps            # List containers
docker stop <id>     # Stop container
```

---

## 📁 Project Structure

```
.
├── app.py                      # Dash app
├── main.py                     # FastAPI backend
├── train.py                    # Model training
├── models.py                   # Model definitions
├── utils/
│   └── data_loader.py          # Data loading logic
├── eda.py                      # EDA & SHAP visualizations
├── configs/                    # Model config files (YAML)
├── results/                    # Saved models, metrics & encoder
├── static/                     # Static images, sample files
├── requirements.txt            # Python dependencies
└── Dockerfile                  # Docker build config
```

---

## 📥 Sample Test File

Download sample input file here: [static/Sample_Test_Data.csv](static/Sample_Test_Data.csv)

---

## 🧪 API Endpoint

Once running, the prediction API is available at:

```http
POST /batch_predict
```

**Payload example:**

```json
{
  "model_name": "RandomForest",
  "columns": ["col1", "col2", ..., "colN"],
  "rows": [["val1", "val2", ..., "valN"], ...]
}
```
