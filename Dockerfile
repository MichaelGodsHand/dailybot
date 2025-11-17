FROM python:3.11-slim

# Install uv
RUN pip install uv

WORKDIR /app

# Copy only dependency files first for caching
COPY pyproject.toml .
COPY uv.lock . 2>/dev/null || true

# Install dependencies from pyproject.toml
RUN uv sync

# Copy the rest of the project (server.py, etc.)
COPY . .

EXPOSE 8080

CMD ["uv", "run", "server.py"]
