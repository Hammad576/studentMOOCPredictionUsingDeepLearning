# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set working directory
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port 5000 for Flask
EXPOSE 5000

# Run Flask in debug mode for live reloading
CMD ["python3", "app.py"]
