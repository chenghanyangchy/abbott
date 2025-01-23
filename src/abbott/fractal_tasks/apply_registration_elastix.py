# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Calculates translation for 2D image-based registration"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable

import anndata as ad
import dask.array as da
import fsspec
import numpy as np
import zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.ngff.zarr_utils import load_NgffWellMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    is_standard_roi_table,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks._zarr_utils import (
    _get_matching_ref_acquisition_path_heuristic,
    _update_well_metadata,
)
from fractal_tasks_core.utils import (
    _get_table_path_dict,
    _split_well_path_image_path,
)
from pydantic import validate_call

from abbott.io.conversions import to_itk, to_numpy
from abbott.registration.itk_elastix import apply_transform, load_parameter_files

logger = logging.getLogger(__name__)


@validate_call
def apply_registration_elastix(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    roi_table: str,
    reference_acquisition: int = 0,
    overwrite_input: bool = True,
    overwrite_output: bool = True,
):
    """Apply registration to images by using a registered ROI table

    This task consists of 4 parts:

    1. Mask all regions in images that are not available in the
    registered ROI table and store each acquisition aligned to the
    reference_acquisition (by looping over ROIs).
    2. Do the same for all label images.
    3. Copy all tables from the non-aligned image to the aligned image
    (currently only works well if the only tables are well & FOV ROI tables
    (registered and original). Not implemented for measurement tables and
    other ROI tables).
    4. Clean up: Delete the old, non-aligned image and rename the new,
    aligned image to take over its place.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        roi_table: Name of the ROI table for which registrations
            have been calculated using the Compute Registration Elastix task.
            Examples: `FOV_ROI_table` => loop over the field of views,
            `well_ROI_table` => process the whole well as one image.
        reference_acquisition: Which acquisition to register against. Uses the
            OME-NGFF HCS well metadata acquisition keys to find the reference
            acquisition.
        overwrite_input: Whether the old image data should be replaced with the
            newly registered image data. Currently only implemented for
            `overwrite_input=True`.
        overwrite_output: Whether pre-existing registered images (which will
            be named "zarr_url" + _registered) should be overwritten by the
            task.

    """
    logger.info(zarr_url)
    logger.info(
        f"Running `apply_registration_to_image` on {zarr_url=}, "
        f"{roi_table=} and {reference_acquisition=}. "
        f"Using {overwrite_input=}"
    )

    well_url, old_img_path = _split_well_path_image_path(zarr_url)
    suffix = "registered"
    new_zarr_url = f"{well_url}/{zarr_url.split('/')[-1]}_{suffix}"
    # Get the zarr_url for the reference acquisition
    acq_dict = load_NgffWellMeta(well_url).get_acquisition_paths()
    logger.info(load_NgffWellMeta(well_url))
    logger.info(acq_dict)
    if reference_acquisition not in acq_dict:
        raise ValueError(
            f"{reference_acquisition=} was not one of the available "
            f"acquisitions in {acq_dict=} for well {well_url}"
        )
    elif len(acq_dict[reference_acquisition]) > 1:
        ref_path = _get_matching_ref_acquisition_path_heuristic(
            acq_dict[reference_acquisition], old_img_path
        )
        logger.warning(
            "Running registration when there are multiple images of the same "
            "acquisition in a well. Using a heuristic to match the reference "
            f"acquisition. Using {ref_path} as the reference image."
        )
    else:
        ref_path = acq_dict[reference_acquisition][0]
    reference_zarr_url = f"{well_url}/{ref_path}"

    # Get acquisition metadata of zarr_url
    curr_acq = get_acquisition_of_zarr_url(well_url, old_img_path)

    # Special handling for the reference acquisition
    # if acq_dict[zarr_url] == reference_zarr_url:
    if curr_acq == reference_acquisition:
        if overwrite_input:
            # If the input is to be overwritten, nothing needs to happen. The
            # reference acquisition stays as is and due to the output type set
            # in the image list, the type of that OME-Zarr is updated
            return
        else:
            # If the input is not overwritten, a copy of the reference
            # OME-Zarr image needs to be created which has the new name & new
            # metadata. It contains the same data as the original reference
            # image.
            generate_copy_of_reference_acquisition(
                zarr_url=zarr_url,
                new_zarr_url=new_zarr_url,
                overwrite=overwrite_output,
            )
            image_list_updates = dict(
                image_list_updates=[dict(zarr_url=new_zarr_url, origin=zarr_url)]
            )
            # Update the metadata of the the well
            well_url, new_img_path = _split_well_path_image_path(new_zarr_url)
            try:
                _update_well_metadata(
                    well_url=well_url,
                    old_image_path=old_img_path,
                    new_image_path=new_img_path,
                )
            except ValueError:
                logger.warning(f"{new_zarr_url} was already listed in well metadata")
            return image_list_updates

    ROI_table_acq = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")

    ngff_image_meta = load_NgffImageMeta(zarr_url)
    coarsening_xy = ngff_image_meta.coarsening_xy
    num_levels = ngff_image_meta.num_levels

    ####################
    # Process images
    ####################
    logger.info("Write the registered Zarr image to disk")
    write_registered_zarr(
        zarr_url=zarr_url,
        new_zarr_url=new_zarr_url,
        ROI_table=ROI_table_acq,
        roi_table_name=roi_table,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        aggregation_function=np.mean,
        overwrite=overwrite_output,
    )

    ####################
    # Process labels
    ####################
    # FIXME: Test if the registration methods are robust for labels. Very
    # likely, it will not always maintain the label values correctly.
    try:
        labels_group = zarr.open_group(f"{zarr_url}/labels", "r")
        label_list = labels_group.attrs["labels"]
        logger.warning(
            "Applying registration to labels. This hasn't been tested well and "
            "may have issues with creating new labels or other errors when "
            "writing the label image."
        )
    except (zarr.errors.GroupNotFoundError, KeyError):
        label_list = []

    if label_list:
        logger.info(f"Processing the label images: {label_list}")
        labels_group = zarr.group(f"{new_zarr_url}/labels")
        labels_group.attrs["labels"] = label_list

        for label in label_list:
            write_registered_zarr(
                zarr_url=f"{zarr_url}/labels/{label}",
                new_zarr_url=f"{new_zarr_url}/labels/{label}",
                ROI_table=ROI_table_acq,
                roi_table_name=roi_table,
                num_levels=num_levels,
                coarsening_xy=coarsening_xy,
                aggregation_function=np.max,
                overwrite=overwrite_output,
            )

    ####################
    # Copy tables
    # 1. Copy all standard ROI tables from the reference acquisition.
    # 2. Copy all tables that aren't standard ROI tables from the given
    # acquisition.
    ####################
    table_dict_reference = _get_table_path_dict(reference_zarr_url)
    table_dict_component = _get_table_path_dict(zarr_url)

    table_dict = {}
    # Define which table should get copied:
    for table in table_dict_reference:
        if is_standard_roi_table(table):
            table_dict[table] = table_dict_reference[table]
    for table in table_dict_component:
        if not is_standard_roi_table(table):
            if reference_zarr_url != zarr_url:
                logger.warning(
                    f"{zarr_url} contained a table that is not a standard "
                    "ROI table. The `Apply Registration To Image task` is "
                    "best used before additional tables are generated. It "
                    f"will copy the {table} from this acquisition without "
                    "applying any transformations. This will work well if "
                    f"{table} contains measurements. But if {table} is a "
                    "custom ROI table coming from another task, the "
                    "transformation is not applied and it will not match "
                    "with the registered image anymore."
                )
            table_dict[table] = table_dict_component[table]

    if table_dict:
        logger.info(f"Processing the tables: {table_dict}")
        new_image_group = zarr.group(new_zarr_url)

        for table in table_dict.keys():
            logger.info(f"Copying table: {table}")
            # Get the relevant metadata of the Zarr table & add it
            # See issue #516 for the need for this workaround
            max_retries = 20
            sleep_time = 5
            current_round = 0
            while current_round < max_retries:
                try:
                    old_table_group = zarr.open_group(table_dict[table], mode="r")
                    current_round = max_retries
                except zarr.errors.GroupNotFoundError:
                    logger.debug(
                        f"Table {table} not found in attempt {current_round}. "
                        f"Waiting {sleep_time} seconds before trying again."
                    )
                    current_round += 1
                    time.sleep(sleep_time)
            # Write the Zarr table
            curr_table = ad.read_zarr(table_dict[table])
            write_table(
                new_image_group,
                table,
                curr_table,
                table_attrs=old_table_group.attrs.asdict(),
                overwrite=True,
            )

    ####################
    # Clean up Zarr file
    ####################
    if overwrite_input:
        logger.info("Replace original zarr image with the newly created Zarr image")
        # Potential for race conditions: Every acquisition reads the
        # reference acquisition, but the reference acquisition also gets
        # modified
        # See issue #516 for the details
        os.rename(zarr_url, f"{zarr_url}_tmp")
        os.rename(new_zarr_url, zarr_url)
        shutil.rmtree(f"{zarr_url}_tmp")
        image_list_updates = dict(image_list_updates=[dict(zarr_url=zarr_url)])
    else:
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=new_zarr_url, origin=zarr_url)]
        )
        # Update the metadata of the the well
        well_url, new_img_path = _split_well_path_image_path(new_zarr_url)
        try:
            _update_well_metadata(
                well_url=well_url,
                old_image_path=old_img_path,
                new_image_path=new_img_path,
            )
        except ValueError:
            logger.warning(f"{new_zarr_url} was already listed in well metadata")

    return image_list_updates


def write_registered_zarr(
    zarr_url: str,
    new_zarr_url: str,
    ROI_table: ad.AnnData,
    roi_table_name: str,
    num_levels: int,
    coarsening_xy: int = 2,
    aggregation_function: Callable = np.mean,
    overwrite: bool = True,
):
    """Write registered zarr array based on ROI tables

    This function loads the image or label data from a zarr array based on the
    ROI bounding-box coordinates and stores them into a new zarr array.
    The new Zarr array has the same shape as the original array, but will have
    0s where the ROI tables don't specify loading of the image data.
    The ROIs loaded from `list_indices` will be written into the
    `list_indices_ref` position, thus performing translational registration if
    the two lists of ROI indices vary.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be used as
            the basis for the new OME-Zarr image.
        new_zarr_url: Path or url to the new OME-Zarr image to be written
        ROI_table: Fractal ROI table for the component
        roi_table_name: Name of the ROI table that the registration was
            calculated on. Used to load the correct registration files.
        num_levels: Number of pyramid layers to be created (argument of
            `build_pyramid`).
        coarsening_xy: Coarsening factor between pyramid levels
        aggregation_function: Function to be used when downsampling (argument
            of `build_pyramid`).
        overwrite: Whether an existing zarr at new_zarr_url should be
            overwritten.

    """
    # Read pixel sizes from Zarr attributes
    level = 0  # FIXME: Expose this to the user
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    pxl_sizes_zyx_full_res = ngff_image_meta.get_pixel_sizes_zyx(level=0)

    # Create list of indices for 3D ROIs
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx_full_res,
    )

    old_image_group = zarr.open_group(zarr_url, mode="r")
    old_ngff_image_meta = load_NgffImageMeta(zarr_url)
    new_image_group = zarr.group(new_zarr_url)
    new_image_group.attrs.put(old_image_group.attrs.asdict())

    # Loop over all channels. For each channel, write full-res image data.
    data_array = da.from_zarr(old_image_group[str(level)])

    new_zarr_array = zarr.create(
        shape=data_array.shape,
        chunks=data_array.chunksize,
        dtype=data_array.dtype,
        store=zarr.storage.FSStore(f"{new_zarr_url}/0"),
        overwrite=overwrite,
        dimension_separator="/",
    )

    # TODO: Add sanity checks on the 2 ROI tables:
    # 1. The number of ROIs need to match
    # 2. The size of the ROIs need to match
    # (otherwise, we can't assign them to the reference regions)
    # ROI_table_ref vs ROI_table_acq
    for i, roi_indices in enumerate(list_indices):
        # reference_region = convert_indices_to_regions(list_indices_ref[i])
        region = convert_indices_to_regions(roi_indices)

        fn_pattern = f"{roi_table_name}_roi_{roi_indices}_t*.txt"

        # FIXME: Improve sorting to always achieve correct order (above 9 items)
        parameter_path = Path(zarr_url) / "registration"
        parameter_files = sorted(parameter_path.glob(fn_pattern))
        parameter_object = load_parameter_files([str(x) for x in parameter_files])

        num_channels = data_array.shape[0]
        # Loop over channels
        axes_list = old_ngff_image_meta.axes_names

        if axes_list == ["c", "z", "y", "x"]:
            for ind_ch in range(num_channels):
                logger.info(f"Processing ROI index {i}, channel {ind_ch}")
                # Define region
                channel_region = (slice(ind_ch, ind_ch + 1), *region)
                itk_img = to_itk(
                    np.squeeze(
                        load_region(
                            data_zyx=data_array[ind_ch], region=region, compute=True
                        )
                    ),
                    scale=tuple(pxl_sizes_zyx),
                )

                parameter_object = adapt_itk_params(
                    parameter_object=parameter_object,
                    itk_img=itk_img,
                )
                registered_roi = apply_transform(
                    itk_img,
                    parameter_object,
                )
                # Write to disk
                img = np.expand_dims(to_numpy(registered_roi), 0)

                # FIXME: Can I still use this kind of writing it? Or do I first
                # need to prepare the whole dask array? If so, memory implication
                # or explicit delayed call?
                da.array(img).to_zarr(
                    url=new_zarr_array,
                    region=channel_region,
                    compute=True,
                )

        elif axes_list == ["z", "y", "x"]:
            # Define region
            itk_img = to_itk(
                load_region(data_zyx=data_array[ind_ch], region=region, compute=True),
                scale=tuple(pxl_sizes_zyx),
            )
            parameter_object = adapt_itk_params(
                parameter_object=parameter_object,
                itk_img=itk_img,
            )
            registered_roi = apply_transform(
                itk_img,
                parameter_object,
            )
            # Write to disk
            img = to_numpy(registered_roi)

            # FIXME: Can I still use this kind of writing it? Or do I first
            # need to prepare the whole dask array? If so, memory implication
            # or explicit delayed call?
            da.array(img).to_zarr(
                url=new_zarr_array,
                region=region,
                compute=True,
            )
        elif axes_list == ["c", "y", "x"]:
            # TODO: Implement cyx case (based on looping over xy case)
            raise NotImplementedError(
                "`write_registered_zarr` has not been implemented for "
                f"a zarr with {axes_list=}"
            )
        elif axes_list == ["y", "x"]:
            # TODO: Implement yx case
            raise NotImplementedError(
                "`write_registered_zarr` has not been implemented for "
                f"a zarr with {axes_list=}"
            )
        else:
            raise NotImplementedError(
                "`write_registered_zarr` has not been implemented for "
                f"a zarr with {axes_list=}"
            )

    # Starting from on-disk highest-resolution data, build and write to
    # disk a pyramid of coarser levels
    build_pyramid(
        zarrurl=new_zarr_url,
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=data_array.chunksize,
        aggregation_function=aggregation_function,
    )


def adapt_itk_params(parameter_object, itk_img):
    """Updates spacing & size settings in the parameter object

    This is needed to address https://github.com/pelkmanslab/abbott/issues/10
    This ensures that applying the transformation will output an image in the
    input resolution (instead of the transform resolution)

    Args:
        parameter_object: ITK parameter object
        itk_img: ITK image that will be registered

    """
    for i in range(parameter_object.GetNumberOfParameterMaps()):
        itk_spacing = tuple([str(x) for x in itk_img.GetSpacing()])
        itk_size = tuple([str(x) for x in itk_img.GetRequestedRegion().GetSize()])
        parameter_object.SetParameter(i, "Spacing", itk_spacing)
        parameter_object.SetParameter(i, "Size", itk_size)
    return parameter_object


def generate_copy_of_reference_acquisition(
    zarr_url: str,
    new_zarr_url: str,
    overwrite: bool = True,
):
    """Generate a copy of an existing OME-Zarr with all its components

    Args:
        zarr_url: Path to the existing zarr image
        new_zarr_url: Path to the to be created zarr image
        overwrite: Whether to overwrite a preexisting new_zarr_url
    """
    # Get filesystem and paths for source and destination
    source_fs, source_path = fsspec.core.url_to_fs(zarr_url)
    dest_fs, dest_path = fsspec.core.url_to_fs(new_zarr_url)

    # Check if the source exists
    if not source_fs.exists(source_path):
        raise FileNotFoundError(f"The source Zarr URL '{zarr_url}' does not exist.")

    # Handle overwrite option
    if dest_fs.exists(dest_path):
        if overwrite:
            dest_fs.delete(dest_path, recursive=True)
        else:
            raise FileExistsError(
                f"The destination Zarr URL '{new_zarr_url}' already exists."
            )

    # Copy the source to the destination
    source_fs.copy(source_path, dest_path, recursive=True)

    # Verify the copied Zarr structure
    try:
        zarr.open_group(source_fs.get_mapper(source_path), mode="r")
        zarr.open_group(dest_fs.get_mapper(dest_path), mode="r")
    except Exception as e:
        raise RuntimeError(f"Failed to verify the copied Zarr structure: {e}") from e


def get_acquisition_of_zarr_url(well_url, image_name):
    """Get the acquisition of a given zarr_url

    Args:
        well_url: Url of the HCS plate well
        image_name: Name of the acquisition image
    """
    well_meta = load_NgffWellMeta(well_url)
    for image in well_meta.well.images:
        if image.path == image_name:
            return image.acquisition


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=apply_registration_elastix,
        logger_name=logger.name,
    )
