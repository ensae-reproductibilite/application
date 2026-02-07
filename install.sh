#!/bin/bash

apt-get -y update && apt-get -y install wget

curl -LsSf https://astral.sh/uv/install.sh | sh

PATH="/home/onyxia/.local/bin:${PATH}"

uv pip install -r pyproject.toml

uv run main.py1
