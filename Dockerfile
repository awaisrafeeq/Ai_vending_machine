# Use a specific Python version for reproducibility
FROM python:3.12.3-slim

# Install system dependencies (fix for PortAudio error)
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your source code
COPY . .

# Run the app using Uvicorn, binding to Render's $PORT
CMD ["uvicorn", "filehandle_databasess_copy:app", "--host", "0.0.0.0", "--port", "$PORT"]
