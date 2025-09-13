FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY data_preprocessing/ ./data_preprocessing/
COPY .gitignore .
COPY LICENSE .
COPY .github/ ./github/

ENTRYPOINT ["python", "-m", "data_preprocessing.main"]