FROM python:3.9-slim

#
WORKDIR /

#
COPY ./ml_conformer_generator /ml_conformer_generator

#
RUN pip install --upgrade pip
RUN pip install -r /ml_conformer_generator/package_requirements.txt

#
CMD ["python", "/ml_conformer_generator/api_server.py"]