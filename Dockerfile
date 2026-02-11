FROM python:3.13-slim
WORKDIR /usr/src/app

# Prevent Python creating .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make entrypoint executable
#RUN chmod +x /usr/src/app/entrypoint.sh

EXPOSE 8000

# Default command (can be overridden in compose)
#CMD ["/usr/src/app/entrypoint.sh"]
