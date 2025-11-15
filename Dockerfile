# Use Google's ML-ready base image (avoids Docker Hub issues)
FROM gcr.io/deeplearning-platform-release/base-cpu

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip and install all requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Cloud Run exposed port
EXPOSE 8080

# Start server using Gunicorn mapped to Flask instance `app` inside app.py
CMD ["gunicorn", "--timeout", "300", "-b", "0.0.0.0:8080", "app:app"]
