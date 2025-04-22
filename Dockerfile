FROM python:3.12-slim

WORKDIR /app

# Copy requirements file and install dependencies
COPY st-requirements.txt .
RUN pip install --no-cache-dir -r st-requirements.txt

# Copy the application code
COPY main.py .

# Copy the examples directory
COPY examples ./examples

# Copy the scripts directory
COPY scripts ./scripts

# Create models directory and copy the model file
RUN mkdir -p models
COPY models/demucs_finetuned.pt ./models/

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]
