# OTREP-X PRIME Core Container
FROM python:3.10-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /opt/otrep-x

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application user and directory
RUN useradd -m -d $APP_HOME -s /bin/bash otrep_user
WORKDIR $APP_HOME

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ $APP_HOME/src
COPY config/ $APP_HOME/config

# Security hardening
RUN chown -R otrep_user:otrep_user $APP_HOME && \
    chmod -R 750 $APP_HOME

USER otrep_user

# Default command
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "src.main:app"]
