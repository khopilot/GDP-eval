# Multi-stage build for GDP Evaluation Framework
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libicu-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libicu67 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 gdpval && \
    mkdir -p /app/data /app/outputs /app/logs /app/.cache && \
    chown -R gdpval:gdpval /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/gdpval/.local

# Copy application code
COPY --chown=gdpval:gdpval . .

# Set Python path
ENV PATH=/home/gdpval/.local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH \
    PYTHONUNBUFFERED=1

# Switch to non-root user
USER gdpval

# Expose ports
EXPOSE 7860 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "demo.py"]