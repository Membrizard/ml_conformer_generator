FROM python:3.9-slim

#
WORKDIR /

#
COPY ./ml_conformer_generator /ml_conformer_generator

# Install RDKit Dependencies
RUN apt-get update && \
    apt-get install -y \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

#
RUN pip install --upgrade pip
RUN pip install -r /ml_conformer_generator/generator_requirements.txt

#
CMD ["python", "/ml_conformer_generator/api_service.py"]