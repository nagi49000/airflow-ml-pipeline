from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import requests
import json
import logging


# fixed vars used in helper functions
raw_samples_filename = "/tmp/raw-samples.json"
test_samples_filename = "/tmp/test-samples.json"
model_filename = "/tmp/lin-reg-model.pickle"


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


def get_trained_model(x, y):
    from sklearn.linear_model import LinearRegression
    logging.info(f"training lin reg on {len(y)} samples")
    lin_reg = LinearRegression().fit(x, y)
    logging.info(f"lin reg trained with coefficient {lin_reg.coef_} and intercept {lin_reg.intercept_}")
    return lin_reg


def evaluate_lin_reg_model(lin_reg):
    x_train, y_train = get_x_matrix_y_vector_from_json(raw_samples_filename)
    x_test, y_test = get_x_matrix_y_vector_from_json(test_samples_filename)
    r2_train = lin_reg.score(x_train, y_train)
    r2_test = lin_reg.score(x_test, y_test)
    logging.info(f"training R^2 score = {r2_train}")
    logging.info(f"test R^2 score = {r2_test}")


# Python callables used in Airflow DAG task definitions
def ingest_data(n_samples_raw=1000, n_samples_test=200):
    get_samples(n_samples_raw, raw_samples_filename)
    get_samples(n_samples_test, test_samples_filename)


def validate_data():
    check_samples_json_file(raw_samples_filename)
    check_samples_json_file(test_samples_filename)


def train():
    logging.info(f"training lin reg on data in {raw_samples_filename}")
    x, y = get_x_matrix_y_vector_from_json(raw_samples_filename)
    reg = get_trained_model(x, y)

    import pickle
    with open(model_filename, "wb") as f:
        pickle.dump(reg, f)
    logging.info(f"wrote lin reg model to {model_filename}")


def evaluate():
    logging.info(f"evaluating model {model_filename} with data {test_samples_filename}")
    import pickle
    with open(model_filename, "rb") as f:
        reg = pickle.load(f)
    evaluate_lin_reg_model(reg)


# Define some arguments for our DAG
default_args = {
    'owner': 'alice',
    'depends_on_past': False,
    'start_date': days_ago(0),
    'retries': 1,
    'retry_delay': timedelta(seconds=20)
}

# Instantiate our DAG
dag = DAG(
    'alpha_orchestrated_experiment',
    default_args=default_args,
    description='A ML orchestrated experiment',
    schedule_interval=timedelta(days=1),
)

with dag:
    data_ingestion_task = PythonOperator(
        task_id='data_ingestion',
        python_callable=ingest_data
    )

    data_validation_task = PythonOperator(
        task_id='data_validation',
        python_callable=validate_data
    )

    # data_preparation_task = PythonOperator(
    #    task_id='data_preparation',
    #    python_callable=prepare_data
    # )

    model_training_task = PythonOperator(
        task_id='model_training',
        python_callable=train
    )

    model_evaluation_task = PythonOperator(
        task_id='model_evaluation',
        python_callable=evaluate
    )

    """
    model_deployment_task = PythonOperator(
        task_id='model_deployment',
        python_callable=deploy
    )
 
    data_ingestion_task >> data_validation_task >> data_preparation_task >> model_training_task >> model_evaluation_task >> model_deployment_task
    """
    data_ingestion_task >> data_validation_task >> model_training_task >> model_evaluation_task
