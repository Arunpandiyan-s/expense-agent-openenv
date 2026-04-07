FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TASK="easy"

ENTRYPOINT ["python", "inference.py"]
CMD ["--task", "easy"]
