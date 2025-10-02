# Pull official Python image
FROM python:3.11.1-slim

# Set working directory inside the container
WORKDIR /app

# Environment variables
# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1  

# Ensure logs appear in real-time (no buffering)
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc libpq-dev
RUN pip install -r requirements.txt

# Copy project files into container
COPY . .

