FROM python:3.10-slim-buster

ENV MLFLOW_HOME=${HOME}/mlflow

WORKDIR ${HOME}/mlflow

COPY requirements.txt requirements.txt

RUN apt-get update && \
    apt-get install -y curl && \
    pip install -r requirements.txt && \
    groupadd --system mlflow && \
    useradd -g mlflow --system mlflow && \
    chown -R mlflow:mlflow ${MLFLOW_HOME} && \
    chmod -R 755 ${MLFLOW_HOME}

USER mlflow

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
