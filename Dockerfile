# Base image with Python
FROM python:3.11-slim

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY uv.lock .    # optional, copy only if you have a lock file
COPY server.py .
COPY src ./src     # if your code lives inside src/
COPY . .

# Install dependencies (this reads pyproject.toml)
RUN uv sync

# Expose port (optional)
EXPOSE 8080

# Run the server with uv
CMD ["uv", "run", "server.py"]
