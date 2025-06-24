# Use official Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Copy project files to the container
COPY . .

# Install system dependencies (for pandas, openpyxl, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose ports (Dash runs on 8050, FastAPI on 8000)
EXPOSE 8050
EXPOSE 8000

# Start FastAPI and Dash together
CMD ["bash", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & python app.py"]