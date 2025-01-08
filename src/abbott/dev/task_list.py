"""Contains the list of tasks available to fractal."""

from fractal_tasks_core.dev.task_models import ParallelTask

TASK_LIST = [
    ParallelTask(
        name="Compute Registration (elastix)",
        executable="fractal_tasks/compute_registration_elastix.py",
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing"],
        docs_info="file:docs_info/calculate_registration.md",
    ),
    ParallelTask(
        name="Apply Registration (elastix)",
        executable="fractal_tasks/apply_registration_elastix.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing"],
        docs_info="file:docs_info/apply_registration.md",
    ),
]
