FROM python:3.8-slim

# Create and change to the app directory.
WORKDIR /usr/src/app

# Copy requirements.txt to the container image.
COPY requirements.txt ./

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy local code to the container image.
COPY . .

# Expose port 5001 for the application.
EXPOSE 5001

# Run the web service on container startup.
CMD ["python", "app.py"]
