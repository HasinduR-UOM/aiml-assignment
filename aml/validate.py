"""Offline schema check of every Azure ML YAML in aml/ (no Azure login needed).

    python -m aml.validate
"""
from pathlib import Path

from azure.ai.ml import load_compute, load_environment, load_job, load_online_deployment, load_online_endpoint
from azure.ai.ml.entities import JobSchedule


def load_schedule(source):
    import yaml
    data = yaml.safe_load(Path(source).read_text())
    return JobSchedule._load(data=data, yaml_path=source)

HERE = Path(__file__).parent
CHECKS = [
    ("environment.yml", load_environment), ("compute.yml", load_compute), ("pipeline.yml", load_job),
    ("endpoint.yml", load_online_endpoint), ("deployment.yml", load_online_deployment),
    ("monitor-job.yml", load_job), ("monitor-schedule.yml", load_schedule),
]

if __name__ == "__main__":
    for name, loader in CHECKS:
        obj = loader(source=HERE / name)
        print(f"OK  {name:22s} -> {type(obj).__name__}")
