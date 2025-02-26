FROM python:3.9-slim

#
WORKDIR /

#
COPY ./mol_edit /mol_edit

RUN pip install --upgrade pip
RUN pip install -r /mol_edit/mol_edit_requirements.txt

#
CMD ["python", "/mol_edit/api_service.py"]