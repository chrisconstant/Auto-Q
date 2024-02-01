# Use a slimmer Python base image
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements.txt first for caching
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining files
COPY . .

# Expose port
EXPOSE 5000

# Set environment variable
ENV FLASK_APP=app.py

# Create a directory for the SQLite database file and set permissions
RUN chmod 777 /app/instance

# Use Gunicorn as the WSGI server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
