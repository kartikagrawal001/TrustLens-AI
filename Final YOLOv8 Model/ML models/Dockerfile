FROM python:3.10-slim

# HuggingFace Spaces runs containers as non-root user uid=1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system libraries required by ultralytics / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app
COPY app.py .

# Copy your trained model weights
# You must have best_v2.pt in this same folder when you push
COPY best_v2.pt ./best_v2.pt

RUN chown -R appuser:appuser /app
USER appuser

# HuggingFace Spaces REQUIRES port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
