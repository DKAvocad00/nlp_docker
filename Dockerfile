FROM python:3.10

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

CMD ["uvicorn", "prediction_endpoints:app", "--host", "0.0.0.0", "--port", "80"]