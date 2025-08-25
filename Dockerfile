FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

COPY requirements.txt /app/
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget git curl \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download fr_core_news_sm \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /app
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
