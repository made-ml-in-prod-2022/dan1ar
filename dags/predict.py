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
MODEL_PATH = "/data/models/{{ var.value.model_date }}"
PREDICT_PATH = "/data/predicted/{{ ds }}"

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(15),
) as dag:
    data_sensor = FileSensor(
        task_id="data_sensor",
        filepath="/opt/airflow/data/raw/{{ ds }}/data.csv"
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir %s --output-dir %s" % (RAW_PATH, PROCESS_PATH),
        task_id="preprocess_data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_PATH, target="/data", type="bind")]
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir %s --model-dir %s --output-dir %s" % (PROCESS_PATH, MODEL_PATH, PREDICT_PATH),
        task_id="predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_PATH, target="/data", type="bind")]
    )

    data_sensor >> preprocess >> predict

    

