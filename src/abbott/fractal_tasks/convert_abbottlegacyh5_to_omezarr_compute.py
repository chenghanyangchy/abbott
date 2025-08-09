# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Ruth Hornbachner <ruth.hornbachner@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""This task converts abbott-legacy H5 files to OME-Zarr."""

import logging
from pathlib import Path
from typing import Optional

import dask.array as da
import pandas as pd
from fractal_tasks_core.cellvoyager.metadata import (
    parse_yokogawa_metadata,
)
from ngio import RoiPixels, open_ome_zarr_container
from ngio.images.ome_zarr_container import create_empty_ome_zarr
from ngio.tables import RoiTable
from ngio.utils._errors import NgioFileExistsError
from pydantic import Field, validate_call

from abbott.fractal_tasks.converter.io_models import (
    ConverterMultiplexingAcquisition,
    ConverterOMEZarrBuilderParams,
    CustomWavelengthInputModel,
    InitArgsCellVoyagerH5toOMEZarr,
)
from abbott.fractal_tasks.converter.task_utils import (
    extract_ROI_coordinates,
    extract_ROIs_from_h5_files,
    find_chunk_shape,
    find_dtype,
    find_shape,
    h5_load,
)

logger = logging.getLogger(__name__)


def convert_single_h5_to_ome(
    zarr_url: str,
    files_well: list[str],
    level: int,
    acquisition_id: str,
    acquisition_params: ConverterMultiplexingAcquisition,
    wavelengths: dict[int, str],
    ome_zarr_parameters: ConverterOMEZarrBuilderParams,
    metadata: pd.DataFrame,
    masking_label: Optional[str] = None,
    overwrite: bool = True,
):
    """Abbott legacy H5 to OME-Zarr converter task.

    Args:
        zarr_url: Output path to save the OME-Zarr file of the form
            `zarr_dir/plate_name/row/column/cycle`.
        files_well: Path to the H5 files per well to be converted.
        level: The level of the image to convert. Default is 0.
        acquisition_id: The multiplexed acquisition cycle.
        acquisition_params: Defines theacquisition by providing allowed
            image and label channels.
        wavelengths: Dictionary mapping wavelength IDs to their OME-Zarr equivalents.
        ome_zarr_parameters: Parameters for the OME-Zarr builder.
        metadata: Metadata DataFrame containing site metadata.
        masking_label: Optional label for masking ROI e.g. `embryo`.
        overwrite: Whether to overwrite existing OME-Zarr data. Default is True.
    """
    logger.info(f"Converting {files_well} to OME-Zarr at {zarr_url}")
    zarr_url = zarr_url.rstrip("/")

    # Extract FOV ROIs
    file_roi_dict, metadata = extract_ROIs_from_h5_files(
        files_well=files_well,
        metadata=metadata,
    )

    # Start extracting image data from the H5 files
    files_dict = {}
    h5_handles = []
    for file in files_well:
        imgs_dict = {}
        channel_wavelengths = []
        for channel in acquisition_params.allowed_image_channels:
            img, scale, h5_handle = h5_load(
                input_path=file,
                channel=channel,
                level=level,
                cycle=int(acquisition_id),
                img_type="intensity",
            )
            channel_wavelengths.append(wavelengths[channel.wavelength_id])
            channel_label = (
                channel.new_label if channel.new_label is not None else channel.label
            )
            imgs_dict[channel_label] = img

        # Check for label images
        if acquisition_params.allowed_label_channels is not None:
            lbls_dict = {}
            for channel in acquisition_params.allowed_label_channels:
                label_img, _, _ = h5_load(
                    input_path=file,
                    channel=channel,
                    level=level,
                    cycle=int(acquisition_id),
                    img_type="label",
                    h5_handle=h5_handle,
                )
                channel_label = (
                    channel.new_label
                    if channel.new_label is not None
                    else channel.label
                )
                lbls_dict[channel_label] = label_img

            lbl_arrays = list(lbls_dict.values())
            channel_lbl_labels = list(lbls_dict.keys())

        # Extract metadata from the h5_file
        ROI_id = file_roi_dict[file]
        pixel_sizes_zyx_dict = {"z": scale[0], "y": scale[1], "x": scale[2]}
        top_left, bottom_right = extract_ROI_coordinates(
            metadata=metadata,
            ROI=ROI_id,
        )

        # Handle single and multi channel images
        img_arrays = list(imgs_dict.values())
        if len(img_arrays) == 1:
            array = da.expand_dims(img_arrays[0], axis=0)
        else:
            array = da.stack(img_arrays, axis=0)

        shape = array.shape
        channel_labels = list(imgs_dict.keys())
        files_dict[ROI_id] = {
            "array": array,
            "lbl_array": lbl_arrays
            if acquisition_params.allowed_label_channels
            else None,
            "channel_labels": channel_labels,
            "channel_lbl_labels": channel_lbl_labels
            if acquisition_params.allowed_label_channels
            else None,
            "channel_wavelengths": channel_wavelengths,
            "shape": shape,
            "top_left_coords": top_left,
            "bottom_right_coords": bottom_right,
        }

    # Check that all files have the same shape, no channels are missing from files
    shapes = [file_dict["shape"] for file_dict in files_dict.values()]
    if len(set(shapes)) != 1:
        raise ValueError(
            f"All files for {files_well} and acquisition {acquisition_id} "
            "must have the same shape. Check if channels are missing."
        )

    # Get on-disk shape
    on_disk_shape = find_shape(
        bottom_right=[
            file_dict["bottom_right_coords"] for file_dict in files_dict.values()
        ],
        dask_imgs=[file_dict["array"] for file_dict in files_dict.values()],
    )

    # Get chunk shape
    chunk_shape = find_chunk_shape(
        dask_imgs=[file_dict["array"] for file_dict in files_dict.values()],
        max_xy_chunk=ome_zarr_parameters.max_xy_chunk,
        z_chunk=ome_zarr_parameters.z_chunk,
        c_chunk=ome_zarr_parameters.c_chunk,
    )

    # Chunk shape should be smaller or equal to the on disk shape
    chunk_shape = tuple(
        min(c, s) for c, s in zip(chunk_shape, on_disk_shape, strict=True)
    )
    img_dtype = find_dtype(
        dask_imgs=[file_dict["array"] for file_dict in files_dict.values()]
    )

    # Try creating the empty OME-Zarr container
    try:
        ome_zarr_container = create_empty_ome_zarr(
            store=zarr_url,
            shape=on_disk_shape,
            chunks=chunk_shape,
            dtype=img_dtype,
            xy_pixelsize=pixel_sizes_zyx_dict["x"],
            z_spacing=pixel_sizes_zyx_dict["z"],
            levels=ome_zarr_parameters.number_multiscale,
            xy_scaling_factor=ome_zarr_parameters.xy_scaling_factor,
            z_scaling_factor=ome_zarr_parameters.z_scaling_factor,
            channel_labels=files_dict[ROI_id]["channel_labels"],
            channel_wavelengths=files_dict[ROI_id]["channel_wavelengths"],
            axes_names=("c", "z", "y", "x"),
            overwrite=overwrite,
        )

    except NgioFileExistsError:
        logger.info(
            f"OME-Zarr group already exists at {zarr_url}. "
            "If you want to overwrite it, set `overwrite=True`."
        )
        ome_zarr_container = open_ome_zarr_container(zarr_url)
        im_list_types = {"is_3D": ome_zarr_container.is_3d}
        return im_list_types

    # Create the well ROI
    well_roi = ome_zarr_container.build_image_roi_table("Well")
    ome_zarr_container.add_table("well_ROI_table", table=well_roi)

    # Write the images as ROIs in the image
    image = ome_zarr_container.get_image()
    pixel_size = image.pixel_size

    _fov_rois = []
    for i, file_params in files_dict.items():
        # Create the ROI for the file
        # Load the whole file and set the data in the image
        image_data = file_params["array"]
        _, s_z, s_y, s_x = file_params["shape"]
        roi_pix = RoiPixels(
            name=f"FOV_{i}",
            x=int(file_params["top_left_coords"].x),
            y=int(file_params["top_left_coords"].y),
            z=int(file_params["top_left_coords"].z),
            x_length=s_x,
            y_length=s_y,
            z_length=s_z,
        )
        roi = roi_pix.to_roi(pixel_size=pixel_size)
        _fov_rois.append(roi)
        image.set_roi(roi=roi, patch=image_data, axes_order=("c", "z", "y", "x"))

    # Build pyramids
    image.consolidate()
    ome_zarr_container.set_channel_percentiles(start_percentile=1, end_percentile=99.9)
    table = RoiTable(rois=_fov_rois)
    ome_zarr_container.add_table("FOV_ROI_table", table=table)

    # Set labels if available
    if files_dict[ROI_id]["lbl_array"]:
        roi_table = ome_zarr_container.get_table("FOV_ROI_table")
        for i, channel_lbl in enumerate(files_dict[ROI_id]["channel_lbl_labels"]):
            label = ome_zarr_container.derive_label(
                name=channel_lbl, overwrite=overwrite
            )
            # For each ROI, set the label data
            for roi in roi_table.rois():
                roi_id = roi.name.split("_")[-1]
                label_array_roi = files_dict[int(roi_id)]["lbl_array"][i]
                label.set_roi(
                    roi=roi, patch=label_array_roi, axes_order=("z", "y", "x")
                )

            label.consolidate()

    for h5_handle in h5_handles:
        h5_handle.close()

    # Build masking roi table if masking label is provided
    if masking_label is not None:
        channel_lbl_labels = files_dict[ROI_id]["channel_lbl_labels"]
        if channel_lbl_labels is not None and masking_label in channel_lbl_labels:
            try:
                masking_roi_table = ome_zarr_container.build_masking_roi_table(
                    masking_label
                )
                ome_zarr_container.add_table(
                    f"{masking_label}_ROI_table",
                    table=masking_roi_table,
                )
                logger.info(f"Built masking ROI table for label {masking_label}")
            except Exception:
                logger.warning(
                    "Failed to build masking ROI table. "
                    "This might be because the label is not present in the data. "
                )

    logger.info(f"Created OME-Zarr container for {files_well} at {zarr_url}")
    im_list_types = {
        "is_3D": ome_zarr_container.is_3d,
    }
    return im_list_types


@validate_call
def convert_abbottlegacyh5_to_omezarr_compute(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: InitArgsCellVoyagerH5toOMEZarr,
    # Core parameters
    level: int = 0,
    wavelengths: CustomWavelengthInputModel = Field(
        title="Wavelengths", default=CustomWavelengthInputModel()
    ),
    axes_names: str = "ZYX",
    ome_zarr_parameters: ConverterOMEZarrBuilderParams = Field(
        title="OME-Zarr Parameters", default=ConverterOMEZarrBuilderParams()
    ),
    masking_label: Optional[str] = None,
):
    """Abbott legacy H5 to OME-Zarr converter task.

    Args:
        zarr_url: Output path to save the OME-Zarr file of the form
            `zarr_dir/plate_name/row/column/`.
        init_args: Initialization arguments passed from init task.
        input_path: Input path to the H5 file, or a folder containing H5 files.
        level: The level of the image to convert. Default is 0.
        wavelengths: Wavelength conversion dictionary mapping.
        axes_names: The layout of the image data. Currently only implemented for 'ZYX'.
        ome_zarr_parameters (OMEZarrBuilderParams): Parameters for the OME-Zarr builder.
        masking_label: Optional label for masking ROI e.g. `embryo`.
    """
    logger.info(f"Converting abbott legacy H5 files to OME-Zarr for {zarr_url}")
    logger.info(f"For axes: {axes_names} and level {level}")

    if axes_names != "ZYX":
        raise ValueError(
            f"Unsupported axes names {axes_names}. "
            "Currently only 'ZYX' is supported for abbott legacy H5 to "
            "OME-Zarr conversion."
        )

    wavelength_conversion_dict = {
        wavelength.wavelength_abbott_legacy: wavelength.wavelength_omezarr
        for wavelength in wavelengths.wavelengths
    }

    # Group files by well
    files = init_args.input_files
    files_well = [file for file in files if init_args.well_ID in Path(file).stem]

    # Get acquisition metadata
    site_metadata, _ = parse_yokogawa_metadata(
        mrf_path=init_args.mrf_path,
        mlf_path=init_args.mlf_path,
    )

    acquisition_id = Path(zarr_url).stem

    im_list_types = convert_single_h5_to_ome(
        zarr_url=zarr_url,
        files_well=files_well,
        level=level,
        acquisition_id=acquisition_id,
        acquisition_params=init_args.acquisition,
        wavelengths=wavelength_conversion_dict,
        ome_zarr_parameters=ome_zarr_parameters,
        metadata=site_metadata,
        masking_label=masking_label,
        overwrite=init_args.overwrite,
    )

    logger.info(f"Succesfully converted {files_well} to {zarr_url}")

    plate_attributes = {
        "well": f"{init_args.well_ID}",
        "plate": f"{init_args.plate_path}",
        "acquisition": str(acquisition_id),
    }

    image_update = {
        "zarr_url": zarr_url,
        "types": im_list_types,
        "attributes": plate_attributes,
    }

    return {"image_list_updates": [image_update]}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=convert_abbottlegacyh5_to_omezarr_compute,
        logger_name=logger.name,
    )
