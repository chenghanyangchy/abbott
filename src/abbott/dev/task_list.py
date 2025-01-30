"""Contains the list of tasks available to fractal."""

from fractal_tasks_core.dev.task_models import CompoundTask, ParallelTask

TASK_LIST = [
    CompoundTask(
        name="Compute Registration (elastix)",
        executable_init="fractal_tasks/init_registration_hcs.py",
        executable="fractal_tasks/compute_registration_elastix.py",
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing"],
        docs_info="file:docs_info/calculate_registration.md",
    ),
    ParallelTask(
        name="Apply Registration (elastix)",
        input_types=dict(registered=False),
        executable="fractal_tasks/apply_registration_elastix.py",
        output_types=dict(registered=True),
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing"],
        docs_info="file:docs_info/apply_registration.md",
    ),
    CompoundTask(
        name="Compute Registration per ROI (elastix)",
        executable_init="fractal_tasks/init_registration_hcs.py",
        executable="fractal_tasks/compute_registration_elastix_per_ROI.py",
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing"],
        docs_info="file:docs_info/calculate_registration.md",
    ),
    ParallelTask(
        name="Apply Registration per ROI (elastix)",
        input_types=dict(registered=False),
        executable="fractal_tasks/apply_registration_elastix_per_ROI.py",
        output_types=dict(registered=True),
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing"],
        docs_info="file:docs_info/apply_registration.md",
    ),
]
