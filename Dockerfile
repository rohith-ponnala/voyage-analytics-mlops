# Base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy all files to container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir flask joblib numpy scikit-learn

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
