FROM python:3.10-slim-buster AS builder

WORKDIR ${HOME}/test

COPY requirements.txt requirements.txt
COPY api/ ./api

RUN pip install -r requirements.txt && \
    pip install pytest flake8 pytest-cov && \
    python -m flake8 && \
    python -m pytest --cov=api/src

FROM python:3.10-slim-buster

ENV APP_HOME=${HOME}/app

WORKDIR ${APP_HOME}

COPY --from=builder ${HOME}/test/requirements.txt requirements.txt
COPY --from=builder ${HOME}/test/api/src/api ./api

RUN apt-get update && \
    apt-get install -y curl && \
    pip install -r requirements.txt && \
    groupadd --system app && \
    useradd -g app --system app && \
    chown -R app:app ${APP_HOME} && \
    chmod -R 755 ${APP_HOME}

USER app

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api.api_app:app", "--bind", "0.0.0.0:6780", "--log-level", "INFO"]
