FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir --no-deps . && useradd --create-home appuser
COPY schedule_change_dataset.csv ./
COPY data ./data
RUN flightbridge train --output /app/artifacts && flightbridge train-intents --output /app/artifacts
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"
CMD ["uvicorn", "flightbridge.api:app", "--host", "0.0.0.0", "--port", "8000"]
