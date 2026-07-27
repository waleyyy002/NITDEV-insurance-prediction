[3:11 pm, 27/07/2026] Machine Learning Ndubuisi: fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
jinja2==3.1.2
pycaret==1.0.0
pandas==1.1.5
numpy==1.19.5
scipy==1.5.4
scikit-learn==0.22.1
joblib==0.17.0
[3:11 pm, 27/07/2026] Machine Learning Ndubuisi: # ---- Dockerfile for the Insurance Charge Predictor (FastAPI app) ----
#
# MLOps note: this image pins EVERY dependency version explicitly and
# installs them all in one step. That single install layer is the one
# source of truth for versions - there's no separate requirements.txt
# fighting with manual pip installs, which is what caused the original
# scikit-learn version mismatch (0.23.2 requested vs 0.24.2 actually
# installed) that broke pycaret's pickled KBinsDiscretizer at runtime.

# 1. Base image: pycaret==1.0.0 is a 2020-era package built for
#    Python 3.6-3.8, so we pin to Python 3.8 rather than a newer version.
FROM python:3.8-slim

# 2. Working directory inside the container.
WORKDIR /app

# 3. OS-level build tools needed to compile ML/scientific packages
#    (numpy/scipy/pycaret/xgboost/lightgbm) if no prebuilt wheel exists
#    for this platform. Installed once, removed after in the same layer
#    to keep the final image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only requirements.txt first so Docker can cache this layer -
#    rebuilds skip re-downloading packages unless this file changes.
COPY requirements.txt .

# 5. Single install step - every version pinned here, matching
#    requirements.txt exactly. scikit-learn MUST stay at 0.23.2 to match
#    what pycaret==1.0.0 expects and what the pickled model was trained
#    with (deployment_28042020). --no-deps on pycaret avoids it pulling
#    in pandas-profiling==2.3.0, which conflicts with newer pandas.
RUN pip install --no-cache-dir --no-deps pycaret==1.0.0 && \
    pip install --no-cache-dir \
    pandas==1.1.5 \
    numpy==1.19.5 \
    scipy==1.5.4 \
    scikit-learn==0.22.1 \
    joblib==0.17.0 \
    fastapi==0.104.1 \
    "uvicorn[standard]==0.24.0" \
    python-multipart==0.0.6 \
    jinja2==3.1.2 \
    ipywidgets \
    pyod \
    xgboost==0.90 \
    yellowbrick==1.0.1 \
    umap-learn \
    lightgbm \
    datefinder

# 6. Copy the rest of the application code (changes often, so placed
#    after the slow dependency layer to maximize cache hits).
COPY . .

# 7. Document the port the app listens on (metadata only - actual
#    port mapping happens via the hosting platform or docker run -p).
EXPOSE 8000

# 8. Start command. --host 0.0.0.0 is required so the server accepts
#    connections from outside the container.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]