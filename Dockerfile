# ---- Dockerfile for the Insurance Charge Predictor (FastAPI app) ----
#
# MLOps note for beginners: a Dockerfile is a recipe for building a
# "container image" - a self-contained snapshot of an OS + Python +
# your dependencies + your code. Anyone who runs this image gets the
# EXACT same environment you tested in, which is why Docker is the
# standard way to ship ML apps: it eliminates "works on my machine"
# problems caused by mismatched Python/library versions (a very common
# issue with older ML libraries like pycaret==1.0.0).

# 1. Base image: start from an official, minimal Python image instead of
#    a full OS. "slim" has just enough OS packages to run Python, which
#    keeps the final image smaller and faster to pull/deploy.
#    pycaret==1.0.0 is a 2020-era package built for Python 3.6-3.8, so we
#    pin the base image to Python 3.8 rather than a newer version.
FROM python:3.8-slim

# 2. Set the working directory inside the container. Every command below
#    (COPY, RUN, CMD) now runs relative to /app inside the container's
#    filesystem - it does not touch your real machine.
WORKDIR /app

# 3. Install OS-level build tools needed to compile some ML/scientific
#    Python packages (e.g. numpy/scipy/pycaret) from source if no
#    pre-built wheel is available for this platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY the requirements file first, then install dependencies.
#    Why not copy everything at once? Docker caches each instruction as a
#    "layer". As long as requirements.txt doesn't change, Docker reuses
#    the cached "pip install" layer on future builds instead of
#    re-downloading every package - this makes rebuilds much faster
#    whenever you only change app.py.
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# 5. Now copy the rest of the application code (this layer changes often,
#    so it's placed after the slow dependency-install layer).
COPY . .

# 6. Document which port the app listens on. This doesn't actually
#    publish the port - it's metadata for humans/tools; the real
#    port mapping happens with `docker run -p` or the hosting platform.
EXPOSE 8000

# 7. The command that runs when the container starts.
#    --host 0.0.0.0 is required so the server accepts connections from
#    outside the container, not just from inside it (127.0.0.1 would be
#    invisible to the outside world).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
