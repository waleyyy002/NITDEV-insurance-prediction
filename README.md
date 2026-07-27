# Insurance Charge Prediction

A FastAPI web service for predicting medical insurance charges based on customer attributes.

The app serves:
- a browser-based form UI (`/`) for entering insurance applicant data
- a JSON prediction API (`/predict_api`)
- a health check endpoint (`/health`)
- interactive API docs at `/docs`

The model is loaded at startup from `deployment_28042020` and used for inference on incoming requests.

## Project structure

- `app.py` – FastAPI application, routes, model loading, and prediction logic.
- `requirements.txt` – Python dependencies for the service.
- `Dockerfile` – container build instructions.
- `Procfile` – command used by hosting platforms like Render.
- `static/` – CSS and other static assets.
- `templates/` – HTML templates.
- `deployment_28042020` – serialized PyCaret regression model loaded by `app.py`.

## Requirements

- Python 3.8+
- Docker (for containerized execution)

## Local setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell
# or
source .venv/bin/activate  # macOS/Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app locally:

```bash
python app.py
```

4. Open the app in a browser:

- `http://localhost:8000` for the form UI
- `http://localhost:8000/docs` for the interactive API docs
- `http://localhost:8000/health` for the health check

## Running with Docker

Build the image:

```bash
docker build -t insurance-predictor .
```

Run the container:

```bash
docker run -d -p 8000:8000 --name insurance-predictor insurance-predictor
```

Then visit `http://localhost:8000`.

## API endpoints

- `GET /` — render the prediction form
- `POST /predict` — submit form data and render prediction result
- `POST /predict_api` — JSON API for programmatic use
- `GET /health` — health check and model-loaded status

### Example JSON request

```bash
curl -X POST http://localhost:8000/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "sex": "female",
    "bmi": 28.5,
    "children": 2,
    "smoker": "no",
    "region": "southeast"
  }'
```

## Notes

- The model input order must match the training schema: `age`, `sex`, `bmi`, `children`, `smoker`, `region`.
- The app uses PyCaret to load and predict from the serialized model.
- If deploying to Render or another cloud provider, make sure the app listens on the expected port and that the model file is included in the deployment bundle.

## Deployment

This repository includes a Dockerfile and Procfile for deployment.

- `Dockerfile` builds a container image for the FastAPI app.
- `Procfile` defines the runtime command for platforms like Render.

If using Render, configure the service to use Docker and expose port `8000`.
