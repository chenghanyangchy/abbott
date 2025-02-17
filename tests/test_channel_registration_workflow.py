import shutil
from pathlib import Path

import pytest
import zarr
from devtools import debug

from abbott.fractal_tasks.apply_channel_registration_elastix import (
    apply_channel_registration_elastix,
)
from abbott.fractal_tasks.compute_channel_registration_elastix import (
    compute_channel_registration_elastix,
)


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path, zenodo_zarr: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    dest_dir = (tmp_path / "registration_data").as_posix()
    debug(zenodo_zarr, dest_dir)
    shutil.copytree(zenodo_zarr, dest_dir)
    return dest_dir


def test_registration_workflow(test_data_dir):
    parameter_files = [
        str(Path(__file__).parent / "data/params_similarity_level1.txt"),
    ]
    # Task-specific arguments
    roi_table = "FOV_ROI_table"
    level = 0
    reference_wavelength = "A01_C01"
    zarr_url = f"{test_data_dir}/B/03/0"

    compute_channel_registration_elastix(
        zarr_url=zarr_url,
        reference_wavelength=reference_wavelength,
        roi_table=roi_table,
        lower_rescale_quantile=0.0,
        upper_rescale_quantile=0.99,
        parameter_files=parameter_files,
        level=level,
    )

    # Test zarr_url that needs to be registered
    apply_channel_registration_elastix(
        zarr_url=zarr_url,
        roi_table=roi_table,
        reference_wavelength=reference_wavelength,
        overwrite_input=False,
    )
    new_zarr_url = f"{zarr_url}_channels_registered"
    zarr.open_group(new_zarr_url, mode="r")

    # Pre-existing output can be overwritten
    apply_channel_registration_elastix(
        zarr_url=zarr_url,
        roi_table=roi_table,
        reference_wavelength=reference_wavelength,
        overwrite_input=False,
        overwrite_output=True,
    )

    apply_channel_registration_elastix(
        zarr_url=zarr_url,
        roi_table=roi_table,
        reference_wavelength=reference_wavelength,
        overwrite_input=True,
    )


def test_registration_workflow_varying_levels(test_data_dir):
    parameter_files = [str(Path(__file__).parent / "data/params_similarity_level1.txt")]
    # Task-specific arguments
    roi_table = "FOV_ROI_table"
    level = 1
    reference_wavelength = "A01_C01"
    zarr_url = f"{test_data_dir}/B/03/0"
    print(zarr_url)

    compute_channel_registration_elastix(
        zarr_url=zarr_url,
        reference_wavelength=reference_wavelength,
        roi_table=roi_table,
        lower_rescale_quantile=0.0,
        upper_rescale_quantile=0.99,
        parameter_files=parameter_files,
        level=level,
    )

    apply_channel_registration_elastix(
        zarr_url=zarr_url,
        roi_table=roi_table,
        reference_wavelength=reference_wavelength,
        overwrite_input=False,
    )
    new_zarr_url = f"{zarr_url}_channels_registered"
    zarr.open_group(new_zarr_url, mode="r")
