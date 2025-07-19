FROM python:3.11-slim AS builder

WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd -r mcp && \
    useradd -r -g mcp -d /app -s /bin/bash mcp && \
    chown -R mcp:mcp /app

USER mcp

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# Run FastAPI server with uvicorn using shell form for variable substitution
CMD uvicorn main:app --host 0.0.0.0 --port $PORT