FROM python:3.9-slim

#
WORKDIR /code

#
COPY ./ml_conformer_generator /code/api_server


#
RUN pip install --upgrade pip
RUN pip install -r /code/api_server/service_requirements.txt


#
CMD ["python", "api_server/api_service.py"]