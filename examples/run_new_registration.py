"""Example to run abbott registration workflows."""

from abbott.fractal_tasks.apply_registration_elastix import apply_registration_elastix
from abbott.fractal_tasks.compute_registration_elastix import (
    compute_registration_elastix,
)
from abbott.fractal_tasks.init_registration_hcs import init_registration_hcs


def test_registration_workflow():
    """Test abbott registration workflow"""
    # TODO: Make path relative
    # parameter_files = [
    #     "/Users/joel/Documents/Code/abbott/tests/data/params_translation_level0.txt"
    # ]
    parameter_files = [
        "/Users/joel/Documents/Code/abbott/tests/data/params_rigid.txt",
        "/Users/joel/Documents/Code/abbott/tests/data/params_affine.txt",
        "/Users/joel/Documents/Code/abbott/tests/data/bspline_lvl2.txt",
    ]
    # Task-specific arguments
    wavelength_id = "A01_C01"
    roi_table = "FOV_ROI_table"
    level = 0
    reference_acquisition = 19

    # FIXME: Get Zarr_urls based on test_data_dir
    test_data_dir = (
        "/Users/joel/Desktop/20241031_abbott/AssayPlate_Greiner_CELLSTAR655090.zarr"
    )
    zarr_urls = [f"{test_data_dir}/B/02/0", f"{test_data_dir}/B/02/1"]

    parallelization_list = init_registration_hcs(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=reference_acquisition,
    )["parallelization_list"]

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

    # FIXME: Make this run on all zarr_urls once the task supports this
    apply_registration_elastix(
        # Fractal parameters
        zarr_url=zarr_urls[1],
        # Core parameters
        roi_table=roi_table,
        reference_acquisition=reference_acquisition,
        overwrite_input=False,
    )


if __name__ == "__main__":
    test_registration_workflow()
