FROM python:3.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run unprivileged; .cache holds the market-data disk cache (mounted as a volume in compose)
RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/.cache \
    && chown -R agent:agent /app/.cache
USER agent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/login', timeout=4)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
