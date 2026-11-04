FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image ~5x smaller than the default CUDA wheel.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces routes to the port declared as `app_port` (7860).
# Locally the same image can run on 8123: docker run -p 8123:8123 -e PORT=8123 .
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
EXPOSE 7860

# Ingest the bundled sample documents, then serve the UI and API.
CMD ["sh", "-c", "python -m app.ingest && uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
