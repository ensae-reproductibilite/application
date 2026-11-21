FROM python:3.13-slim

# System dependency needed by lightgbm
RUN apt-get -y update && \
    apt-get install -y libgomp1

# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY train.py .
COPY app ./app

CMD ["bash", "-c", "./app/run.sh"]
