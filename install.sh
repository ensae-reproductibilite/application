#!/usr/bin/env bash

# Ensure everything is installed
sudo apt-get -y update
sudo apt-get install -y python3-pip python3-venv curl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restore environment
uv sync
