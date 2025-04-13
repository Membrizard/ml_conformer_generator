FROM python:3.9-slim

#
WORKDIR /

#
COPY ./api_server /api_server

#
RUN pip install --upgrade pip
RUN pip install -r server_requirements.txt

#
CMD ["python", "api_server.py"]