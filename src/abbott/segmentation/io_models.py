# Copyright 2023 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi  <joel.luethi@fmi.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Helper functions for image normalization
"""
import logging
from typing import Optional

from enum import Enum
from pydantic import BaseModel, Field

from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.channels import ChannelNotFoundError
from fractal_tasks_core.channels import get_channel_from_image_zarr
from fractal_tasks_core.channels import OmeroChannel

from abbott.segmentation.stardist_utils import StardistCustomNormalizer

logger = logging.getLogger(__name__)

class StardistModels(Enum):
    """
    Enum for Stardist model names
    """
    VERSATILE_FLUO_2D = "2D_versatile_fluo"
    VERSATILE_HE_2D = "2D_versatile_he"
    PAPER_DSB2018_2D = "2D_paper_dsb2018"
    DEMO_2D = "2D_demo"
    DEMO_3D = "3D_demo"
    
class StardistpretrainedModel(BaseModel):
    """
    Parameters to load a custom pretrained model
    
    Attributes:
        base_fld: Base folder to where custom Stardist models are stored
        pretrained_model_name: Name of the custom model
    """
    base_fld: str
    pretrained_model_name: str
    
class StardistModelParams(BaseModel):
    """
    Advanced Stardist Model Parameters
    
    Attributes:
        prob_thresh: Only consider objects with predicted object 
            probability above this threshold
        nms_thresh: Perform non-maximum suppression (NMS) that 
            considers two objects to be the same when their area/surface 
            overlap exceeds this threshold.
        normalize: 
        scale: Scale the input image internally by a factor and rescale 
            the output accordingly.
        use_gpu: If `False`, always use the CPU; if `True`, use the GPU if
            possible and fall-back to the CPU otherwise.
    """
    
    prob_thresh: Optional[float] = None
    nms_thresh: Optional[float] = None
    normalize: bool = True
    scale: int = 1
    use_gpu: bool = True
    
    
class StardistChannelInputModel(ChannelInputModel):
    """
    Channel input for cellpose with normalization options.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
        normalize: Validator to handle different normalization scenarios for
            Cellpose models
    """

    normalization: StardistCustomNormalizer = Field(
        default_factory=StardistCustomNormalizer
    )

    def get_omero_channel(self, zarr_url) -> OmeroChannel:
        try:
            return get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=self.wavelength_id,
                label=self.label,
            )
        except ChannelNotFoundError as e:
            logger.warning(
                f"Channel with wavelength_id: {self.wavelength_id} "
                f"and label: {self.label} not found, exit from the task.\n"
                f"Original error: {str(e)}"
            )
            return None
