FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV BACKEND_URL=http://backend:8000
ENV LICENSE_CODE=TEST_LICENSE_123
ENV PYTHONUNBUFFERED=1

# Create shared data directory and set permissions
RUN mkdir -p /shared_data && \
    chmod 777 /shared_data && \
    touch /shared_data/shared_data.json && \
    chmod 666 /shared_data/shared_data.json

# Expose the port Streamlit runs on
EXPOSE 8501

# Create a non-root user
RUN useradd -m -u 1000 streamlit
USER streamlit

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 