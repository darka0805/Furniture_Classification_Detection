# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Note: Installing the CPU-only version of PyTorch to keep image size smaller
# If GPU support is needed, remove the --extra-index-url part
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir pandas seaborn matplotlib scikit-learn Pillow

# Copy the current directory contents into the container at /app
COPY . .

# Create directory for mounting data if it doesn't exist
RUN mkdir -p /app/dataset

# Make the inference script executable (optional)
RUN chmod +x inference.py

# Define default command - helpful text
CMD ["python", "inference.py", "--help"]
