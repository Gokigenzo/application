FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer streams
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies required by OpenCV (GUI libraries) and compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package resolution
RUN pip install --no-cache-dir uv

# Copy requirements list
COPY requirements.txt .

# Install dependencies system-wide inside the container using uv
RUN uv pip install --system --no-cache -r requirements.txt

# Copy codebase
COPY . .

# Expose default Streamlit web interface port
EXPOSE 8501

# Run Streamlit dev server
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

