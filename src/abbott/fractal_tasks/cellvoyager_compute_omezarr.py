"""Init registration module for image-based registration based on tasks-core."""

import logging

from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_compute import (
    cellvoyager_to_ome_zarr_compute,
)
from fractal_tasks_core.tasks.io_models import ChunkSizes, InitArgsCellVoyager
from pydantic import Field, validate_call

logger = logging.getLogger(__name__)


@validate_call
def cellvoyager_compute_omezarr(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: InitArgsCellVoyager,
    chunk_sizes: ChunkSizes = Field(default_factory=ChunkSizes),
):
    """Convert Yokogawa output (png, tif) to zarr file.

    This task is run after an init task (typically
    `cellvoyager_to_ome_zarr_init` or
    `cellvoyager_to_ome_zarr_init_multiplex`), and it populates the empty
    OME-Zarr files that were prepared.

    Note that the current task always overwrites existing data. To avoid this
    behavior, set the `overwrite` argument of the init task to `False`.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `create_cellvoyager_ome_zarr_init`.
        chunk_sizes: Used to overwrite the default chunk sizes for the
            OME-Zarr. By default, the task will chunk the same as the
            microscope field of view size, with 10 z planes per chunk.
            For example, that can mean c: 1, z: 10, y: 2160, x:2560
    """
    return cellvoyager_to_ome_zarr_compute(
        zarr_url=zarr_url,
        init_args=init_args,
        chunk_sizes=chunk_sizes,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=cellvoyager_compute_omezarr,
        logger_name=logger.name,
    )
