FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy application files
COPY . .

# ensure entrypoint script has executable permissions
RUN chmod +x entrypoint.sh

# expose streamlit and fastapi ports
EXPOSE 7860 8000

# run entrypoint script
CMD ["/bin/bash", "entrypoint.sh"]