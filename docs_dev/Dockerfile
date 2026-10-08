FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir mkdocs-material

COPY docs/ ./docs
COPY mkdocs.yml ./mkdocs.yml

CMD ["mkdocs", "serve", "-a", "0.0.0.0:8432"]