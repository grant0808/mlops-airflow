from __future__ import annotations

# [START tutorial]
# [START import_module]
import textwrap
from datetime import datetime, timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonVirtualenvOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from tasks.eval import eval
from tasks.is_model_drift import is_model_drift



with DAG(
    "eval-Model",
    default_args={
        "depends_on_past": False,
        # "retries": 1,
        # "retry_delay": timedelta(minutes=5),
    },
    
    description="eval model",
    schedule=timedelta(days=7),
    start_date=datetime(2025, 3, 28),
    catchup=False,
) as dag:
    eval_task = PythonVirtualenvOperator(
        task_id="eval_task",
        requirements=["google-cloud-storage","scikit-learn", "pandas", "numpy", "torch"],
        python_callable=eval
    )

    is_model_drift_task = ShortCircuitOperator(
        task_id="is_model_drift_task",
        python_callable=is_model_drift
    )

    train_trigger_task = TriggerDagRunOperator(
        task_id="train_trigger_task",
        trigger_dag_id="train"
    )

    eval_task >> is_model_drift_task >> train_trigger_task