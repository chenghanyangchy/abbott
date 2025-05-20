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
"""Helper functions for image normalization"""

import logging
from enum import Enum
from typing import Optional

from fractal_tasks_core.channels import (
    ChannelInputModel,
    ChannelNotFoundError,
    OmeroChannel,
    get_channel_from_image_zarr,
)
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from abbott.segmentation.segmentation_utils import (
    SeededSegmentationCustomNormalizer,
    StardistCustomNormalizer,
)

logger = logging.getLogger(__name__)


class StardistModels(Enum):
    """Enum for Stardist model names"""

    VERSATILE_FLUO_2D = "2D_versatile_fluo"
    VERSATILE_HE_2D = "2D_versatile_he"
    PAPER_DSB2018_2D = "2D_paper_dsb2018"
    DEMO_2D = "2D_demo"
    DEMO_3D = "3D_demo"


class StardistpretrainedModel(BaseModel):
    """Parameters to load a custom pretrained model

    Attributes:
        base_fld: Base folder to where custom Stardist models are stored
        pretrained_model_name: Name of the custom model
    """

    base_fld: str
    pretrained_model_name: str


class StardistChannelInputModel(ChannelInputModel):
    """Channel input for Stardist with normalization options.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
        normalization: Validator to handle different normalization scenarios for
            Stardist models
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
                f"Original error: {e!s}"
            )
            return None


class StardistModelParams(BaseModel):
    """Advanced Stardist Model Parameters

    Attributes:
        sparse: If true, aggregate probabilities/distances sparsely during tiled
            prediction to save memory (recommended)
        prob_thresh: Consider only object candidates from pixels with predicted
            object probability above this threshold.
        nms_thresh: Perform non-maximum suppression (NMS) that
            considers two objects to be the same when their area/surface
            overlap exceeds this threshold.
        scale: Scale the input image internally by a tuple of floats and rescale
            the output accordingly. Useful if the Stardist model has been trained
            on images with different scaling. E.g. (z, y, x) = (1.0, 0.5, 0.5).
        n_tiles : Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled. This parameter denotes a
            tuple of the number of tiles for every image axis.
            E.g. (z, y, x) = (2, 4, 4).
        show_tile_progress: Whether to show progress during tiled prediction.
        verbose: Whether to print some info messages.
        predict_kwargs: Keyword arguments for ``predict`` function of Keras model.
        nms_kwargs: Keyword arguments for non-maximum suppression.
    """

    sparse: bool = True
    prob_thresh: Optional[float] = None
    nms_thresh: Optional[float] = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    n_tiles: tuple[int, int, int] = (1, 1, 1)
    show_tile_progress: bool = False
    verbose: bool = False
    predict_kwargs: dict = None
    nms_kwargs: dict = None


class OptionalChannelInputModel(BaseModel):
    """A channel which is specified by either `wavelength_id` or `label` or None.

    This model is similar to `OmeroChannel`, but it is used for
    task-function arguments (and for generating appropriate JSON schemas).

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
    """

    wavelength_id: Optional[str] = None
    label: Optional[str] = None

    @model_validator(mode="after")
    def mutually_exclusive_channel_attributes(self: Self) -> Self:
        """Check that `label` and `wavelength_id` are not set simultaneously."""
        wavelength_id = self.wavelength_id
        label = self.label

        if wavelength_id and label:
            raise ValueError(
                "`wavelength_id` and `label` cannot be both set "
                f"(given {wavelength_id=} and {label=})."
            )
        return self


class SeededSegmentationChannelInputModel(OptionalChannelInputModel):
    """Channel input for seeded segmentation with normalization options.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
        normalize: Validator to handle different normalization scenarios for
            seeded segmentation task.
    """

    normalize: SeededSegmentationCustomNormalizer = Field(
        default_factory=SeededSegmentationCustomNormalizer
    )

    def get_omero_channel(self, zarr_url) -> OmeroChannel:
        try:
            return get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=self.wavelength_id,
                label=self.label,
            )
        except ValueError as e:
            logger.warning(
                f"Channel with wavelength_id: {self.wavelength_id} "
                f"and label: {self.label} not found, exit from the task.\n"
                f"Original error: {e!s}"
            )
            return None


class FilterType(Enum):
    """Enum for image filter types"""

    EROSION = "erosion"
    DILATION = "dilation"
    OPENING = "opening"
    CLOSING = "closing"


class ImageFilter(BaseModel):
    """Advanced Seeded Segmentation Parameters.

    Attributes:
        filter_type: Type of morphological image filter to apply to itk.LabelMap
        filter_value: Value of the filter to apply to itk.LabelMap
    """

    filter_type: Optional[FilterType] = None
    filter_value: Optional[int] = None

    @model_validator(mode="after")
    def mutually_exclusive_channel_attributes(self: Self) -> Self:
        """Check that if `filter_type` is None, `filter_value` is also None."""
        filter_type = self.filter_type
        filter_value = self.filter_value

        if not filter_type and filter_value is not None:
            raise ValueError(
                f"`filter_type` can not be set to None if {filter_value=}."
            )
        return self


class SeededSegmentationParams(ImageFilter):
    """Advanced Seeded Segmentation Parameters.

    Attributes:
        filter_type: Type of morphological image filter to apply to itk.LabelMap
        filter_value: Value of the filter to apply to itk.LabelMap
        filter_radius: Filter radius to use for the seeded segmentation.
        compactness: Parameter for skimage.segmentation.watershed. Higher values
            result in more regularly-shaped watershed basins.
    """

    filter_radius: Optional[int] = None
    compactness: float = 0.0
