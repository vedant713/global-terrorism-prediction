FROM python:3.9-slim

# Install system dependencies
# libgomp1 is required for XGBoost on Linux
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY ../requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose ports for both services (documentation only, compose handles mapping)
EXPOSE 8000
EXPOSE 8501
