# FastAPI Dockerfile (django_backend_dream/dream-fast-api/Dockerfile)
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages (psycopg2, faiss, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8002

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
