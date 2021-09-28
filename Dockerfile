FROM python:3.8.6-buster

COPY pix2pix /pix2pix
COPY api /api
COPY generator.h5 /generator.h5
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install python-multipart

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
