FROM python:3.11-slim

RUN pip install uv

WORKDIR /app

COPY pyproject.toml .
COPY server.py .
COPY . .

RUN uv sync

EXPOSE 8080
CMD ["uv", "run", "server.py"]
