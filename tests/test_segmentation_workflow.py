import shutil
from pathlib import Path

import pytest
from devtools import debug

from abbott.segmentation.io_models import (StardistModels,
                                           StardistChannelInputModel)
from abbott.segmentation.stardist_utils import (StardistCustomNormalizer)
from abbott.fractal_tasks.stardist_segmentation import stardist_segmentation

from fractal_tasks_core.zarr_utils import OverwriteNotAllowedError

@pytest.fixture(scope="function")
def test_data_dir_2d(tmp_path: Path, zenodo_zarr_stardist: list) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    zenodo_zarr_url = zenodo_zarr_stardist[1]
    dest_dir = (tmp_path / "stardist_data").as_posix()
    debug(zenodo_zarr_url, dest_dir)
    shutil.copytree(zenodo_zarr_url, dest_dir)
    return dest_dir


def test_stardist_segmentation_workflow_2d(test_data_dir_2d):
    # Task-specific arguments
    input_ROI_table = "FOV_ROI_table"
    stardist_model = StardistModels.VERSATILE_FLUO_2D
    output_label_name = "nuclei_stardist"
    
    zarr_url = f"{test_data_dir_2d}/B/03/0"
    
    channel = StardistChannelInputModel(
            wavelength_id="A01_C01",
            normalization=StardistCustomNormalizer(),
        )
    
    stardist_segmentation(
        zarr_url=zarr_url,
        level=2,
        channel=channel,
        input_ROI_table=input_ROI_table,
        model_type=stardist_model,
        output_label_name=output_label_name,
        advanced_stardist_model_params=dict(
            prob_thresh=0.1,
            nms_thresh=0.4,
            scale=tuple([1.0, 1.0, 1.0]),
        ),
        overwrite=True,
    )
    
    with pytest.raises(OverwriteNotAllowedError):
        stardist_segmentation(
            zarr_url=zarr_url,
            level=2,
            channel=channel,
            input_ROI_table=input_ROI_table,
            model_type=stardist_model,
            output_label_name=output_label_name,
            advanced_stardist_model_params=dict(
                prob_thresh=0.1,
                nms_thresh=0.4,
                scale=tuple([1.0, 1.0, 1.0]),
            ),
            overwrite=False,
        )


@pytest.fixture(scope="function")
def test_data_dir_3d(tmp_path: Path, zenodo_zarr: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    dest_dir = (tmp_path / "stardist_data").as_posix()
    debug(zenodo_zarr, dest_dir)
    shutil.copytree(zenodo_zarr, dest_dir)
    return dest_dir


def test_stardist_segmentation_workflow_3d(test_data_dir_3d):
    # Task-specific arguments
    input_ROI_table = "FOV_ROI_table"
    stardist_model = StardistModels.DEMO_3D
    output_label_name = "nuclei_stardist"
    
    zarr_url = f"{test_data_dir_3d}/B/03/0"
    
    channel = StardistChannelInputModel(
            wavelength_id="A01_C01",
            normalization=StardistCustomNormalizer(),
        )
    
    advanced_stardist_model_params = dict(prob_thresh=0.5,
                                          nms_thresh=0.4,
                                          scale=tuple([1.0, 1.0, 1.0]))
    
    stardist_segmentation(
        zarr_url=zarr_url,
        level=4,
        channel=channel,
        input_ROI_table=input_ROI_table,
        model_type=stardist_model,
        use_masks=False,
        output_label_name=output_label_name,
        advanced_stardist_model_params=advanced_stardist_model_params,
        overwrite=True,
    )
    
    # test the same function with pretrained-model
    pretrained_model = dict(base_fld=str(Path(__file__).parent / "data/stardist_models/"), 
                            pretrained_model_name="custom_3D")
    
    stardist_segmentation(
        zarr_url=zarr_url,
        level=4,
        use_masks=True,
        channel=channel,
        input_ROI_table=input_ROI_table,
        pretrained_model=pretrained_model,
        output_label_name=output_label_name,
        advanced_stardist_model_params=advanced_stardist_model_params,
        overwrite=True,
    )

    # test stardist segmentation on embryo_ROIs
    input_roi_table_masked = "emb_ROI_table_2_linked"
    
    stardist_segmentation(
        zarr_url=zarr_url,
        level=4,
        use_masks=True,
        channel=channel,
        input_ROI_table=input_roi_table_masked,
        model_type=stardist_model,
        output_label_name=output_label_name,
        advanced_stardist_model_params=advanced_stardist_model_params,
        overwrite=True,
    )




