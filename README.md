# MLOPS_OnlineTraining

# 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Mohsin-Ali-Mirza/MLOPS_OnlineTraining
cd MLOPS_OnlineTraining
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Running the Application

The application consists of two components that need to be run simultaneously:

### Backend (FastAPI)

Open a terminal and run:

```bash
cd app
uvicorn backend:app
```

The backend API will be available at `http://localhost:8000`

### Frontend (Streamlit)

Open a **new terminal**, activate the virtual environment, and run:

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
cd app
streamlit run frontend.py
```

The frontend will be available at `http://localhost:8501`


## 👤 Authors

- Mohsin Ali Mirza
- Waleed Gul
