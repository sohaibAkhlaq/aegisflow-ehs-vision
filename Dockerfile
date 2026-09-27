# AegisFlow EHS - single container: pipeline + API + dashboard.
# Build:  docker build -t aegisflow-ehs .
# Run:    docker compose up   (see docker-compose.yml - handles the .env and volumes)

FROM python:3.11-slim

# System dependencies: OpenCV needs libgl1 and libglib2.0-0 even in "headless" setups,
# and ffmpeg gives OpenCV's video I/O backend more format support at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (separate layer - only re-installs when
# requirements change, not on every source code edit).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional copilot dependencies - comment out if you don't need the AI Copilot layer.
COPY requirements-copilot.txt .
RUN pip install --no-cache-dir -r requirements-copilot.txt

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# data/, outputs/, and the sqlite db live outside the image (see docker-compose.yml's
# volumes) so real footage and generated reports survive a container rebuild.
CMD ["python", "-m", "aegisflow", "serve"]