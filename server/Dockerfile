FROM python:3.9-slim

#
WORKDIR /

#
COPY ./ /server

#
RUN pip install --upgrade pip
RUN pip install -r /server/server_requirements.txt

#
CMD ["python", "/server/api_server.py"]