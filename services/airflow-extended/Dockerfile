FROM apache/airflow:2.5.0-python3.10

# switch to root for installs
USER root

RUN apt-get update && \
    apt-get install -y git

# switch back to non-root USER
USER 50000

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
