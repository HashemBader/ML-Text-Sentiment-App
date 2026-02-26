# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required by text_preprocess
RUN python -m nltk.downloader stopwords wordnet punkt

# Copy application source files
COPY web_app.py .
COPY src/ ./src/
COPY web/ ./web/

# Copy the model into the expected models/ subdirectory
COPY classification_logreg.joblib ./models/classification_logreg.joblib

# Expose the application port
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "web_app:app"]
