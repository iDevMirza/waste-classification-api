
FROM python:3.11-slim

WORKDIR /app

# Python and CPU inference settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=10000

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy dependency list
COPY requirements.txt .

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source and model checkpoint
COPY app/ ./app/
COPY weights/ ./weights/

# Render default web service port
EXPOSE 10000

# Start FastAPI
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
