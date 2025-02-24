FROM python:3.9-slim

#
WORKDIR /code

#
COPY ./ml_conformer_generator /code/ml_conformer_generator


#
RUN pip install --upgrade pip
RUN pip install -r /code/ml_conformer_generator/generator_requirements.txt


#
CMD ["python", "/code/ml_conformer_generator/api_service.py"]