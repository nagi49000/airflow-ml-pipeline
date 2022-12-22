# airflow-ml-pipeline
play area for using airflow to orchestrate data ETL and ml pipeline training

### Airflow set up

The docker set up for airflow is a little involved. The (docker compose file)[./docker-compose.yaml] and instructions to run are pulled from the
[airflow website](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html).

The docker compose file is a little complicated, and will require docker compose v1.29 or above. Instructions for installing
compose as a plugin are linked from the [airflow website](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html),
or can be accessed directly [here](https://docs.docker.com/compose/install/).
The steps described in the [Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) notes are
- set up the folders and airflow user to be the local user
- initialize the airflow database
- start all airflow services

Once everything is up and running, the UI on the webserver can be accessed at http://localhost:8080 with username alice password bob.