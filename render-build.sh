#!/usr/bin/env bash
# exit on error
set -o errexit

# Update apt-get
apt-get update

# Install libreoffice for headless docx → pdf conversion
apt-get install -y --no-install-recommends libreoffice-core libreoffice-writer libreoffice-java-common default-jre-headless

# Install uv and Python dependencies from pyproject.toml
pip install uv
uv sync --frozen


