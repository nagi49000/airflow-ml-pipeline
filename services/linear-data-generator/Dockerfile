FROM python:3.10-slim-buster

ENV APP_HOME=${HOME}/app

WORKDIR ${APP_HOME}

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    groupadd --system app && \
    useradd -g app --system app && \
    chown -R app:app ${APP_HOME} && \
    chmod -R 755 ${APP_HOME}

COPY api/src/api ./api

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api.api_app:app", "--bind", "0.0.0.0:6780", "--log-level", "DEBUG"]