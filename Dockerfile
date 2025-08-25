# Use an official slim Python
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system deps needed for some wheels / runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker layer caching works
COPY requirements.txt /app/

# Upgrade pip/setuptools/wheel and install CPU torch explicitly, then other requirements.
# Installing torch separately via PyTorch CPU index reduces risk of a CUDA wheel being pulled.
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
       "torch>=2.0.0,<3.0" \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download fr_core_news_sm \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy app
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
