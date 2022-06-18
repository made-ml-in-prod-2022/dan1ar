from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

# !!! PATH ON LOCAL MACHINE !!!
MOUNT_PATH = '/Users/danilandreev/Desktop/Python/DataScience/Technopark/MLProd/made-ml-in-prod-2022_homework3/data'

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "generate_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(5),
) as dag:
    generate_task = DockerOperator(
        image="airflow-generate-data",
        command="--output_dir /data/raw/{{ ds }}",
        task_id="generate_data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_PATH, target="/data", type="bind")]
    )