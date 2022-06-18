from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

# !!! PATH ON LOCAL MACHINE !!!
MOUNT_PATH = '/Users/danilandreev/Desktop/Python/DataScience/Technopark/MLProd/made-ml-in-prod-2022_homework3/data'
RAW_PATH = "/data/raw/{{ ds }}"
PROCESS_PATH = "/data/processed/{{ ds }}"
MODEL_PATH = "/data/models/{{ ds }}"

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "train_model",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(15),
) as dag:
    data_sensor = FileSensor(
        task_id="data_sensor",
        filepath="/opt/airflow/data/raw/{{ ds }}/data.csv"
    )

    target_sensor = FileSensor(
        task_id="target_sensor",
        filepath="/opt/airflow/data/raw/{{ ds }}/target.csv"
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir %s --output-dir %s" % (RAW_PATH, PROCESS_PATH),
        task_id="preprocess_data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_PATH, target="/data", type="bind")]
    )

    train_val_split = DockerOperator(
        image="airflow-train-val-splitter",
        command="--input-dir %s --val-size %f" % (PROCESS_PATH, 0.2),
        task_id="train_val_split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_PATH, target="/data", type="bind")]
    )

    train_model = DockerOperator(
        image="airflow-train-model",
        command="--input-dir %s --model-dir %s" % (PROCESS_PATH, MODEL_PATH),
        task_id="train_model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_PATH, target="/data", type="bind")]
    )

    validate_model = DockerOperator(
        image="airflow-validate-model",
        command="--input-dir %s --model-dir %s" % (PROCESS_PATH, MODEL_PATH),
        task_id="validate_model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_PATH, target="/data", type="bind")]
    )

    [data_sensor, target_sensor] >> preprocess >> train_val_split
    train_val_split >> train_model >> validate_model

    

