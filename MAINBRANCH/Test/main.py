FROM python:3.11-alpine
RUN apk add --no-cache gcc musl-dev python3-dev mariadb-dev
WORKDIR /app
COPY . /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "main.py"]
