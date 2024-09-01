# Use the official Python image from the Docker Hub
FROM python:3.11.0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Command to run the Flask application
CMD ["python", "app.py"]

# Expose the port the app runs on
EXPOSE 5000