import shutil
from pathlib import Path

import pytest
import zarr
from devtools import debug

from abbott.fractal_tasks.apply_registration_elastix import apply_registration_elastix
from abbott.fractal_tasks.apply_registration_elastix_per_ROI import (
    apply_registration_elastix_per_ROI,
)
from abbott.fractal_tasks.compute_registration_elastix import (
    compute_registration_elastix,
)
from abbott.fractal_tasks.compute_registration_elastix_per_ROI import (
    compute_registration_elastix_per_ROI,
)
from abbott.fractal_tasks.init_registration_hcs import init_registration_hcs


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
        str(Path(__file__).parent / "data/params_rigid.txt"),
        str(Path(__file__).parent / "data/params_affine.txt"),
        str(Path(__file__).parent / "data/bspline_lvl2.txt"),
    ]
    # Task-specific arguments
    wavelength_id = "A01_C01"
    roi_table = "FOV_ROI_table"
    level = 0
    reference_acquisition = 19
    zarr_urls = [f"{test_data_dir}/B/02/0", f"{test_data_dir}/B/02/1"]

    parallelization_list = init_registration_hcs(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=reference_acquisition,
    )["parallelization_list"]
    print(parallelization_list)

    for param in parallelization_list:
        compute_registration_elastix(
            zarr_url=param["zarr_url"],
            init_args=param["init_args"],
            wavelength_id=wavelength_id,
            roi_table=roi_table,
            lower_rescale_quantile=0.0,
            upper_rescale_quantile=0.99,
            parameter_files=parameter_files,
            level=level,
        )

    # Test zarr_url that needs to be registered
    apply_registration_elastix(
        zarr_url=zarr_urls[1],
        roi_table=roi_table,
        reference_acquisition=reference_acquisition,
        overwrite_input=False,
    )
    new_zarr_url = f"{zarr_urls[1]}_registered"
    zarr.open_group(new_zarr_url, mode="r")

    # Test reference zarr
    apply_registration_elastix(
        zarr_url=zarr_urls[0],
        roi_table=roi_table,
        reference_acquisition=reference_acquisition,
        overwrite_input=False,
    )
    new_zarr_url = f"{zarr_urls[0]}_registered"
    zarr.open_group(new_zarr_url, mode="r")

    # Test overwrite output False on reference image
    with pytest.raises(FileExistsError):
        apply_registration_elastix(
            zarr_url=zarr_urls[0],
            roi_table=roi_table,
            reference_acquisition=reference_acquisition,
            overwrite_input=False,
            overwrite_output=False,
        )

    # Test overwrite output False on non reference image
    with pytest.raises(zarr.errors.ContainsArrayError):
        apply_registration_elastix(
            zarr_url=zarr_urls[1],
            roi_table=roi_table,
            reference_acquisition=reference_acquisition,
            overwrite_input=False,
            overwrite_output=False,
        )

    # Pre-existing output can be overwritten
    for zarr_url in zarr_urls:
        apply_registration_elastix(
            zarr_url=zarr_url,
            roi_table=roi_table,
            reference_acquisition=reference_acquisition,
            overwrite_input=False,
            overwrite_output=True,
        )

    for zarr_url in zarr_urls:
        apply_registration_elastix(
            zarr_url=zarr_url,
            roi_table=roi_table,
            reference_acquisition=reference_acquisition,
            overwrite_input=True,
        )


def test_registration_workflow_varying_levels(test_data_dir):
    parameter_files = [
        str(Path(__file__).parent / "data/params_rigid.txt"),
        str(Path(__file__).parent / "data/params_affine.txt"),
        str(Path(__file__).parent / "data/bspline_lvl2.txt"),
    ]
    # Task-specific arguments
    wavelength_id = "A01_C01"
    roi_table = "FOV_ROI_table"
    level = 1
    reference_acquisition = 19
    zarr_urls = [f"{test_data_dir}/B/02/0", f"{test_data_dir}/B/02/1"]

    parallelization_list = init_registration_hcs(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=reference_acquisition,
    )["parallelization_list"]
    print(parallelization_list)

    for param in parallelization_list:
        compute_registration_elastix(
            zarr_url=param["zarr_url"],
            init_args=param["init_args"],
            wavelength_id=wavelength_id,
            roi_table=roi_table,
            lower_rescale_quantile=0.0,
            upper_rescale_quantile=0.99,
            parameter_files=parameter_files,
            level=level,
        )

    # Test zarr_url that needs to be registered
    apply_registration_elastix(
        zarr_url=zarr_urls[1],
        roi_table=roi_table,
        reference_acquisition=reference_acquisition,
        overwrite_input=False,
    )
    new_zarr_url = f"{zarr_urls[1]}_registered"
    zarr.open_group(new_zarr_url, mode="r")


# TODO: Add test data for these tasks
def test_registration_workflow_ROI():
    parameter_files = [
        str(Path(__file__).parent / "data/params_rigid.txt"),
        str(Path(__file__).parent / "data/params_affine.txt"),
        str(Path(__file__).parent / "data/bspline_lvl2.txt"),
    ]
    # Task-specific arguments
    wavelength_id = "A01_C01"
    label_name = "emb_linked"
    roi_table = "emb_ROI_table_linked"
    level = 2
    reference_acquisition = 0
    fld = """/Volumes/active/rhornb/fractal/74_tiny_mouseembryo_dataset/
    397__data_active_rhornb_fractal_testing_embryo/"""

    zarr_urls = [
        f"{fld}/AssayPlate_Greiner_#655090.zarr/E/02/0",
        f"{fld}/AssayPlate_Greiner_#655090.zarr/E/02/1",
    ]

    parallelization_list = init_registration_hcs(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=reference_acquisition,
    )["parallelization_list"]
    print(parallelization_list)

    for param in parallelization_list:
        compute_registration_elastix_per_ROI(
            zarr_url=param["zarr_url"],
            init_args=param["init_args"],
            wavelength_id=wavelength_id,
            lower_rescale_quantile=0.0,
            upper_rescale_quantile=0.99,
            label_name=label_name,
            roi_table=roi_table,
            parameter_files=parameter_files,
            level=level,
        )

    # Test zarr_url that needs to be registered
    apply_registration_elastix_per_ROI(
        zarr_url=zarr_urls[1],
        roi_table=roi_table,
        label_name=label_name,
        reference_acquisition=reference_acquisition,
        overwrite_input=False,
    )
    new_zarr_url = f"{zarr_urls[1]}_registered"
    zarr.open_group(new_zarr_url, mode="r")
