import shutil
from pathlib import Path

import pytest
from devtools import debug

from abbott.fractal_tasks.init_registration_hcs import init_registration_hcs


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    # FIXME: Get the data from zenodo
    source_dir = (
        "/Users/joel/Desktop/20241031_abbott/AssayPlate_Greiner_CELLSTAR655090.zarr"
    )
    dest_dir = (tmp_path / "registration_data").as_posix()
    debug(source_dir, dest_dir)
    shutil.copytree(source_dir, dest_dir)
    return dest_dir


def test_registration_workflow(test_data_dir):
    # TODO: Make path relative
    # parameter_files = [
    #     "/Users/joel/Documents/Code/abbott/tests/data/params_translation_level0.txt"
    # ]
    # # Task-specific arguments
    # wavelength_id = "A01_C01"
    # roi_table = "FOV_ROI_table"
    # level = 0
    # intensity_normalization = False

    zarr_urls = []
    print(test_data_dir)

    parallelization_list = init_registration_hcs(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=19,
    )

    print(parallelization_list)

    # compute_registration_elastix(
    #     zarr_url=zarr_url,
    #     wavelength_id=wavelength_id,
    #     roi_table=roi_table,
    #     reference_cycle=reference_cycle,
    #     parameter_files=parameter_files,
    #     level=level,
    #     intensity_normalization=intensity_normalization,
    # )

    # apply_registration_elastix(
    #     input_paths=input_paths,
    #     output_path=output_path,
    #     component=component,
    #     metadata=metadata,
    #     roi_table=roi_table,
    #     reference_cycle=reference_cycle,
    #     overwrite_input=False,
    #     registration_folder="transforms",
    # )
