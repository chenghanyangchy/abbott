# Copyright 2023 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi  <joel.luethi@fmi.ch>
# Ruth Hornbachner <ruth.hornbachner@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Helper functions to run stardist segmentation task
"""
import logging
from typing import Optional, Literal

from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from fractal_tasks_core.tasks.cellpose_utils import normalized_img
from csbdeep.utils import normalize


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
        model_name: Name of the custom model
    """
    
    base_fld: str
    pretrained_model_name: str
    

class StardistCustomNormalizer(BaseModel):
    """
    Validator to handle different normalization scenarios for Stardist models

    If `type="default"`, then Stardist default normalization is
    used and no other parameters can be specified.
    If `type="no_normalization"`, then no normalization is used and no
    other parameters can be specified.
    If `type="custom"`, then either percentiles or explicit integer
    bounds can be applied.

    Attributes:
        type:
            One of `default` (Stardist default normalization), `custom`
            (using the other custom parameters) or `no_normalization`.
        lower_percentile: Specify a custom lower-bound percentile for rescaling
            as a float value between 0 and 100. Set to 1 to run the same as
            default). You can only specify percentiles or bounds, not both.
        upper_percentile: Specify a custom upper-bound percentile for rescaling
            as a float value between 0 and 100. Set to 99 to run the same as
            default, set to e.g. 99.99 if the default rescaling was too harsh.
            You can only specify percentiles or bounds, not both.
        lower_bound: Explicit lower bound value to rescale the image at.
            Needs to be an integer, e.g. 100.
            You can only specify percentiles or bounds, not both.
        upper_bound: Explicit upper bound value to rescale the image at.
            Needs to be an integer, e.g. 2000.
            You can only specify percentiles or bounds, not both.
    """

    type: Literal["default", "custom", "no_normalization"] = "default"
    lower_percentile: Optional[float] = Field(None, ge=0, le=100)
    upper_percentile: Optional[float] = Field(None, ge=0, le=100)
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None

    # In the future, add an option to allow using precomputed percentiles
    # that are stored in OME-Zarr histograms and use this pydantic model that
    # those histograms actually exist

    @model_validator(mode="after")
    def validate_conditions(self: Self) -> Self:
        # Extract values
        type = self.type
        lower_percentile = self.lower_percentile
        upper_percentile = self.upper_percentile
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        # Verify that custom parameters are only provided when type="custom"
        if type != "custom":
            if lower_percentile is not None:
                raise ValueError(
                    f"Type='{type}' but {lower_percentile=}. "
                    "Hint: set type='custom'."
                )
            if upper_percentile is not None:
                raise ValueError(
                    f"Type='{type}' but {upper_percentile=}. "
                    "Hint: set type='custom'."
                )
            if lower_bound is not None:
                raise ValueError(
                    f"Type='{type}' but {lower_bound=}. "
                    "Hint: set type='custom'."
                )
            if upper_bound is not None:
                raise ValueError(
                    f"Type='{type}' but {upper_bound=}. "
                    "Hint: set type='custom'."
                )

        # The only valid options are:
        # 1. Both percentiles are set and both bounds are unset
        # 2. Both bounds are set and both percentiles are unset
        are_percentiles_set = (
            lower_percentile is not None,
            upper_percentile is not None,
        )
        are_bounds_set = (
            lower_bound is not None,
            upper_bound is not None,
        )
        if len(set(are_percentiles_set)) != 1:
            raise ValueError(
                "Both lower_percentile and upper_percentile must be set "
                "together."
            )
        if len(set(are_bounds_set)) != 1:
            raise ValueError(
                "Both lower_bound and upper_bound must be set together"
            )
        if lower_percentile is not None and lower_bound is not None:
            raise ValueError(
                "You cannot set both explicit bounds and percentile bounds "
                "at the same time. Hint: use only one of the two options."
            )

        return self
    

def _normalize_stardist_channel(
    x: np.ndarray,
    normalization: StardistCustomNormalizer,
) -> np.ndarray:
    """
    Normalize a cellpose input array by channel.

    Args:
        x: 3D numpy array.
        normalization: By default, data is normalized so 0.0=1st percentile and
            1.0=99th percentile of image intensities in each channel.
            This automatic normalization can lead to issues when the image to
            be segmented is very sparse. You can turn off the default
            rescaling. With the "custom" option, you can either provide your
            own rescaling percentiles or fixed rescaling upper and lower
            bound integers.

    """
    # Optionally perform custom normalization
    if normalization.type == "custom":
        x = normalized_img(
            x,
            lower_p=normalization.lower_percentile,
            upper_p=normalization.upper_percentile,
            lower_bound=normalization.lower_bound,
            upper_bound=normalization.upper_bound,
        )
        
    if normalization.type == "default":
        x = normalize(x)
        
    return x




