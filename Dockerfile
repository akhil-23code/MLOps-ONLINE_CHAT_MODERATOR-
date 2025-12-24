# 1. Use a lightweight Python image
FROM python:3.12-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Ensure the app directory is in the python path
ENV PYTHONPATH=/app 

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements first
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- ADDED STEP: Download NLTK data ---
RUN python -m nltk.downloader vader_lexicon

# 7. Copy the rest of your app and the model
COPY . .

# 8. Expose the port FastAPI runs on
EXPOSE 8000

# 9. Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]