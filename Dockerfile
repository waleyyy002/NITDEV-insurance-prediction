# ---- Dockerfile for the Insurance Charge Predictor (FastAPI app) ----
FROM python:3.8-slim-bullseye

WORKDIR /app

# Install OS-level build tools - using bullseye repositories that are still active
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libgomp1 \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]