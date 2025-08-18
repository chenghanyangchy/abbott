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
"""Calculates registration for image-based registration."""

import json
import logging
from pathlib import Path
from typing import Optional

from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from ngio import open_ome_zarr_container
from pydantic import validate_call

from abbott.registration.fractal_helper_tasks import pad_to_max_shape

logger = logging.getLogger(__name__)


@validate_call
def compute_registration_warpfield(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Core parameters
    level: int,
    wavelength_id: str,
    path_to_registration_recipe: Optional[str] = None,
    roi_table: str = "FOV_ROI_table",  # TODO: allow "emb_ROI_table"
    use_masks: bool = False,
    masking_label_name: Optional[str] = None,
) -> None:
    """Calculate warpfield registration based on images

    This task consists of 3 parts:

    1. Loading the images of a given ROI (=> loop over ROIs)
    2. Calculating the transformation for that ROI
    3. Storing the calculated transformation in the ROI table

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        level: Pyramid level of the image to be used for registration.
            Choose `0` to process at full resolution.
        wavelength_id: Wavelength that will be used for image-based
            registration; e.g. `A01_C01` for Yokogawa, `C01` for MD.
        path_to_registration_recipe: Path to the warpfield .yml registration recipe.
            This parameter is optional, if not provided, the default .yml recipe
            will be used.
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. Examples: `FOV_ROI_table` => loop over
            the field of views, `well_ROI_table` => process the whole well as
            one image.
        use_masks:  If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            actually be processed (e.g. running within `embryo_ROI_table`).
        masking_label_name: Optional label for masking ROI e.g. `embryo`.
    """
    try:
        import warpfield
    except ImportError as e:
        raise ImportError(
            "The `compute_registration_warpfield` task requires GPU. "
        ) from e

    logger.info(
        f"Running for {zarr_url=}.\n"
        f"Calculating warpfield registration per {roi_table=} for "
        f"{wavelength_id=}."
    )

    reference_zarr_url = init_args.reference_zarr_url

    # Load channel to register by
    ome_zarr_ref = open_ome_zarr_container(reference_zarr_url)
    channel_index_ref = ome_zarr_ref.image_meta._get_channel_idx_by_wavelength_id(
        wavelength_id
    )

    ome_zarr_mov = open_ome_zarr_container(zarr_url)
    channel_index_align = ome_zarr_mov.image_meta._get_channel_idx_by_wavelength_id(
        wavelength_id
    )

    # Get images for the given level and at highest resolution
    ref_images = ome_zarr_ref.get_image(path=str(level))
    mov_images = ome_zarr_mov.get_image(path=str(level))

    ref_images_full_res = ome_zarr_ref.get_image(path="0")
    mov_images_full_res = ome_zarr_mov.get_image(path="0")

    # Read ROIs
    ref_roi_table = ome_zarr_ref.get_table(roi_table)
    mov_roi_table = ome_zarr_mov.get_table(roi_table)

    # Masked loading checks
    if use_masks:
        if ref_roi_table.type() != "masking_roi_table":
            logger.warning(
                f"ROI table {roi_table} in reference OME-Zarr is not "
                "a masking ROI table. Falling back to use_masks=False."
            )
            use_masks = False
        if masking_label_name is None:
            logger.warning(
                "No masking label provided, but use_masks is True. "
                "Falling back to use_masks=False."
            )
            use_masks = False

        ref_images = ome_zarr_ref.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table,
            path=str(level),
        )

        mov_images = ome_zarr_mov.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table,
            path=str(level),
        )

        # Get also at highest resolution
        ref_images_full_res = ome_zarr_ref.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table,
            path="0",
        )
        mov_images_full_res = ome_zarr_mov.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table,
            path="0",
        )

    logger.info(
        f"Found {len(ref_roi_table.rois())} ROIs in {roi_table=} to be processed."
    )

    # For each acquisition, get the relevant info
    # TODO: Add additional checks on ROIs?
    if len(ref_roi_table.rois()) != len(mov_roi_table.rois()):
        raise ValueError(
            "Registration is only implemented for ROIs that match between the "
            "acquisitions (e.g. well, FOV ROIs). Here, the ROIs in the "
            f"reference acquisitions were {len(ref_roi_table.rois())}, but the "
            f"ROIs in the alignment acquisition were {mov_roi_table.rois()}."
        )

    # Read full-res pixel sizes from zarr attributes
    pxl_sizes_zyx_ref_full_res = ome_zarr_ref.get_image(path="0").pixel_size.zyx
    pxl_sizes_zyx_mov_full_res = ome_zarr_mov.get_image(path="0").pixel_size.zyx

    if pxl_sizes_zyx_ref_full_res != pxl_sizes_zyx_mov_full_res:
        raise ValueError(
            "Pixel sizes need to be equal between acquisitions "
            "for warpfield registration."
        )

    num_ROIs = len(ref_roi_table.rois())
    for i_ROI, ref_roi in enumerate(ref_roi_table.rois()):
        # For masked loading, assumes that i_ROI+1 == label_id of ROI
        mov_roi = mov_roi_table.rois()[i_ROI]
        logger.info(
            f"Now processing ROI {i_ROI+1}/{num_ROIs} " f"for {wavelength_id=}."
        )

        if use_masks:
            img_ref = ref_images.get_roi_masked(
                label=i_ROI + 1,
                c=channel_index_ref,
            ).squeeze()
            img_mov = mov_images.get_roi_masked(
                label=i_ROI + 1,
                c=channel_index_align,
            ).squeeze()

            # Get also full resolution images
            img_ref_full_res = ref_images_full_res.get_roi_masked(
                label=i_ROI + 1,
                c=channel_index_ref,
            ).squeeze()
            img_mov_full_res = mov_images_full_res.get_roi_masked(
                label=i_ROI + 1,
                c=channel_index_align,
            ).squeeze()

            # Pad images to the same shape
            # Calculate maximum dimensions needed
            max_shape = tuple(
                max(r, m) for r, m in zip(img_ref.shape, img_mov.shape, strict=False)
            )
            max_shape_full_res = tuple(
                max(r, m)
                for r, m in zip(
                    img_ref_full_res.shape, img_mov_full_res.shape, strict=False
                )
            )
            img_ref = pad_to_max_shape(img_ref, max_shape)
            img_mov = pad_to_max_shape(img_mov, max_shape)

            img_ref_full_res = pad_to_max_shape(img_ref_full_res, max_shape_full_res)
            img_mov_full_res = pad_to_max_shape(img_mov_full_res, max_shape_full_res)

        else:
            img_ref = ref_images.get_roi(
                roi=ref_roi,
                c=channel_index_ref,
            ).squeeze()
            img_mov = mov_images.get_roi(
                roi=mov_roi,
                c=channel_index_align,
            ).squeeze()

            # Get also full resolution images
            img_ref_full_res = ref_images_full_res.get_roi(
                roi=ref_roi,
                c=channel_index_ref,
            ).squeeze()
            img_mov_full_res = mov_images_full_res.get_roi(
                roi=mov_roi,
                c=channel_index_align,
            ).squeeze()

        ##############
        #  Calculate the transformation
        ##############
        if img_ref.shape != img_mov.shape:
            raise NotImplementedError(
                "This registration is not implemented for ROIs with "
                "different shapes between acquisitions."
            )

        if path_to_registration_recipe is not None:
            try:
                recipe = warpfield.Recipe.from_yaml(path_to_registration_recipe)
            except Exception as e:
                raise ValueError(
                    "Failed to load registration recipe from "
                    f"{path_to_registration_recipe}. "
                    "Please check the file path and format."
                ) from e
        else:
            recipe = warpfield.Recipe.from_yaml("default.yml")

        # Start registration
        _, warp_map, _ = warpfield.register_volumes(img_ref, img_mov, recipe)

        # Write transform parameter files
        # TODO: Add overwrite check (it overwrites by default)
        # FIXME: Figure out where to put files
        fn = Path(zarr_url) / "registration" / (f"{roi_table}_roi_{i_ROI}.json")

        if level > 0:
            downsample_factor = 2 * level
            resize_dict = {
                "warp_field_shape": warp_map.warp_field.shape,
                "block_size": [
                    warp_map.block_size.tolist()[0],
                    warp_map.block_size.tolist()[1] * downsample_factor,
                    warp_map.block_size.tolist()[2] * downsample_factor,
                ],
                "block_stride": [
                    warp_map.block_stride.tolist()[0],
                    warp_map.block_stride.tolist()[1] * downsample_factor,
                    warp_map.block_stride.tolist()[2] * downsample_factor,
                ],
            }
            lvl0_warp_map = warp_map.resize_to(resize_dict)
            lvl0_warp_map.ref_shape = img_ref_full_res.shape
            lvl0_warp_map.mov_shape = img_mov_full_res.shape

            warpfield_dict = {
                "warp_field": lvl0_warp_map.warp_field.tolist(),
                "block_size": lvl0_warp_map.block_size.tolist(),
                "block_stride": lvl0_warp_map.block_stride.tolist(),
                "ref_shape": lvl0_warp_map.ref_shape,
                "mov_shape": lvl0_warp_map.mov_shape,
            }
        else:
            warpfield_dict = {
                "warp_field": warp_map.warp_field.tolist(),
                "block_size": warp_map.block_size.tolist(),
                "block_stride": warp_map.block_stride.tolist(),
                "ref_shape": warp_map.ref_shape,
                "mov_shape": warp_map.mov_shape,
            }
        logger.info(f"{warpfield_dict=}")

        fn.parent.mkdir(exist_ok=True, parents=True)
        with open(fn, "w") as f:
            json.dump(warpfield_dict, f)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=compute_registration_warpfield,
        logger_name=logger.name,
    )
