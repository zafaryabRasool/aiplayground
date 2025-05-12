FROM python:3.12-slim

RUN apt-get update && apt-get install default-libmysqlclient-dev pkg-config build-essential -y

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8080

CMD alembic upgrade head;python app.py
