FROM astral/uv:python3.13-trixie-slim

# System dependency needed by lightgbm
RUN apt-get -y update && \
    apt-get install -y libgomp1

# Install project dependencies (development ones excluded, also for uv run)
ENV UV_NO_DEV=1
COPY pyproject.toml .
RUN uv sync

COPY train.py .
COPY src ./src
CMD ["uv", "run", "train.py"]
