FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

COPY models/ ./models/
COPY dash/ ./dash/
COPY *.py ./
COPY *.sh ./

RUN chmod +x startup.sh
RUN mkdir -p signals

EXPOSE 5000
EXPOSE 5050

CMD ["./startup.sh"]
