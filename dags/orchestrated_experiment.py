from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import requests
import json


def ingest_data(n_samples=1000):
    response = requests.post("http://line-samples:6780/get-samples", json={"n_samples": n_samples})
    assert response.ok
    with open("/tmp/raw-samples.json", "wt") as f:
        json.dump(response.json(), f)


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

    """
    data_validation_task = PythonOperator(
        task_id='data_validation',
        python_callable=validate_data
    )

    data_preparation_task = PythonOperator(
        task_id='data_preparation',
        python_callable=prepare_data
    )

    model_training_task = PythonOperator(
        task_id='model_training',
        python_callable=train
    )

    model_evaluation_task = PythonOperator(
        task_id='model_evaluation',
        python_callable=evaluate
    )

    model_deployment_task = PythonOperator(
        task_id='model_deployment',
        python_callable=deploy
    )
 
    data_ingestion_task >> data_validation_task >> data_preparation_task >> model_training_task >> model_evaluation_task >> model_deployment_task
    """
    data_ingestion_task
