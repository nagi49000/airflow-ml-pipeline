# airflow-ml-pipeline
A play area for using airflow to orchestrate data ETL and ml pipeline training.

### Airflow set up

The docker set up for airflow is a little involved. The [docker compose file](./docker-compose.yaml) and instructions to run are pulled from the
[airflow website](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) - accessed Dec 2022.

The docker compose file is a little complicated, and will require docker compose v1.29 or above. Instructions for installing
compose as a plugin are linked from the [airflow website](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html),
or can be accessed directly [here](https://docs.docker.com/compose/install/).
The steps described in the [Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) notes are
- set up the folders and airflow user to be the local user with
```
mkdir -p ./dags ./logs ./plugins
echo -e "AIRFLOW_UID=$(id -u)" > .env
```
- initialize the airflow database with `docker compose up airflow-init`
- start all airflow services with `docker compose up`

Once everything is up and running, the UI on the webserver can be accessed at http://localhost:8080 with username alice password bob.

To clean up the set up as a 'nuke from orbit' option, use
```
docker compose down --volumes --remove-orphans
```

### Running ML experiments

The custom containers running include a MLFlow server, and a simple API that generates numbers on a line (with some noise), in the x interval \[0, 1\].
There is a simple DAG that takes samples from that API, builds a model, and submits the experiment to MLFlow. The experiment can
be triggered manually.

Once the services are up, the DAGs and triggering thereof can be accessed from the [Airflow webserver UI](http://localhost:8080/), and
the experiment results and storage on the [MLFlow UI](http://localhost:5000/). The random sample generator API also has
[swagger docs](http://localhost:6780/docs).
