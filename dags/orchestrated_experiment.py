from datetime import timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import requests
import json
import logging


# fixed vars used in helper functions
raw_samples_filename = "/tmp/raw-samples.json"
test_samples_filename = "/tmp/test-samples.json"
mlflow_server = "http://mlflow-server:5000"
model_git_url = "https://github.com/nagi49000/airflow-ml-pipeline-alpha-orchestrated-model.git"


# various helper functions for Airflow Python callables
def get_samples(n_samples, filename):
    url = "http://line-samples:6780/get-samples"
    logging.info(f"pulling {n_samples} samples from {url}")
    response = requests.post(url, json={"n_samples": n_samples})
    assert response.ok
    with open(filename, "wt") as f:
        json.dump(response.json(), f)
    logging.info(f"wrote raw samples to {filename}")


def check_samples_json_file(filename):
    logging.info(f"validating raw samples in {filename}")
    with open(filename, "rt") as f:
        j = json.load(f)
    samples = j["samples"]
    for sample in samples:
        assert len(sample) == 2
        assert isinstance(sample[0], float)
        assert isinstance(sample[1], float)


def get_x_matrix_y_vector_from_json(filename):
    import numpy as np
    logging.info(f"reading samples from {filename}")
    with open(filename, "rt") as f:
        j = json.load(f)
    x = np.array([[s[0]] for s in j["samples"]])
    y = np.array([s[1] for s in j["samples"]])
    return x, y


def get_model_and_gitsha(git_url):
    from os import path
    from shutil import rmtree
    from git import Repo
    from random import randint
    from importlib import import_module

    # make up a name for pulling down a repo with the model into the dags folder
    repo_name = f"model_repo_{randint(10000, 99999)}"
    # need to clone into an area that can be picked up on sys.path, such as the DAGs folder
    repo_dir_and_name = path.join("dags", repo_name)
    try:
        logging.info(f"cloning {git_url} to {repo_dir_and_name}")
        repo = Repo.clone_from(git_url, repo_dir_and_name)
        gitsha = repo.head.object.hexsha
        model_module = import_module(f"{repo_name}.python.model.model")  # known path inside repo
        lin_reg = model_module.get_model()  # known function inside repo
    finally:
        rmtree(repo_dir_and_name)  # clean up after ourselves
    return lin_reg, gitsha

def train_and_evaluate_model(git_url):
    lin_reg, gitsha = get_model_and_gitsha(git_url)

    x_train, y_train = get_x_matrix_y_vector_from_json(raw_samples_filename)
    logging.info(f"training lin reg on {len(y_train)} samples")
    lin_reg.fit(x_train, y_train)
    logging.info(f"lin reg trained with coefficient {lin_reg.coef_} and intercept {lin_reg.intercept_}")
    r2_train = lin_reg.score(x_train, y_train)
    logging.info(f"training R^2 score = {r2_train}")

    x_test, y_test = get_x_matrix_y_vector_from_json(test_samples_filename)
    r2_test = lin_reg.score(x_test, y_test)
    logging.info(f"test R^2 score = {r2_test}")

    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(mlflow_server)
    mlflow.set_experiment("alpha_orchestrated_experiment")
    mlflow.log_metrics({
        "training-R2-score": r2_train,
        "test-R2-score": r2_test
    })
    mlflow.log_params({
        "coefficient": lin_reg.coef_,
        "intercept": lin_reg.intercept_,
        "model-url": git_url,
        "model-gitsha": gitsha
    })
    mlflow.sklearn.log_model(lin_reg, "model")
    logging.info(f"sent experiment details to mlflow run {mlflow.active_run().info.run_name}")
    mlflow.end_run()


# Python callables used in Airflow DAG task definitions
def ingest_data(n_samples_raw=1000, n_samples_test=200):
    get_samples(n_samples_raw, raw_samples_filename)
    get_samples(n_samples_test, test_samples_filename)


def validate_data():
    check_samples_json_file(raw_samples_filename)
    check_samples_json_file(test_samples_filename)


def train():
    train_and_evaluate_model(model_git_url)


# Define some arguments for our DAG
default_args = {
    "owner": "alice",
    "depends_on_past": False,
    "start_date": days_ago(0),
    "retries": 1,
    "retry_delay": timedelta(seconds=20)
}

# Instantiate our DAG
dag = DAG(
    "alpha_orchestrated_experiment",
    default_args=default_args,
    description="A ML orchestrated experiment",
    schedule_interval=timedelta(days=1),
)

with dag:
    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=ingest_data
    )

    data_validation_task = PythonOperator(
        task_id="data_validation",
        python_callable=validate_data
    )

    # data_preparation_task = PythonOperator(
    #    task_id="data_preparation",
    #    python_callable=prepare_data
    # )

    model_training_task = PythonOperator(
        task_id="model_training",
        python_callable=train
    )

    # model_evaluation_task = PythonOperator(
    #    task_id="model_evaluation",
    #    python_callable=evaluate
    # )

    # model_deployment_task = PythonOperator(
    #    task_id="model_deployment",
    #    python_callable=deploy
    # )
 
    # data_ingestion_task >> data_validation_task >> data_preparation_task >> model_training_task >> model_evaluation_task >> model_deployment_task

    data_ingestion_task >> data_validation_task >> model_training_task
