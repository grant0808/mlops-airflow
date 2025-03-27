from __future__ import annotations

# [START tutorial]
# [START import_module]
import textwrap
from datetime import datetime, timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator
from task.train import train

with DAG(
    "tutorial",
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    # [END default_args]
    description="train",
    schedule=timedelta(days=30),
    start_date=datetime(2021, 1, 1),
    catchup=False,
) as dag:
    train_task = PythonOperator(
        task_id="train_task",
        requirements=[],
        python_callable=train
    )