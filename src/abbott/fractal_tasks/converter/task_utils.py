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

"""Task utils for abbott H5 legacy to OME-Zarr converter Fractal task"""

import logging
from pathlib import Path
from typing import Optional

import dask.array as da
import h5py
import pandas as pd
from ngio import PixelSize

from abbott.fractal_tasks.converter.io_models import (
    ConverterOmeroChannel,
)
from abbott.fractal_tasks.converter.tile import OriginDict, Point

logger = logging.getLogger(__name__)


def parse_filename(filename: str) -> dict[str, str]:
    """Parse image metadata from filename.

    Args:
        filename: Name of the image.

    Returns:
        Metadata dictionary.
    """
    # Remove extension and folder from filename
    filename = Path(filename).stem

    output = {}

    # Split filename into well + x_coords + y_coords
    filename_fields = filename.split("_")
    if len(filename_fields) < 3:
        raise ValueError(f"{filename} not valid")

    # Assign well
    well = filename_fields[0]
    x_coords = filename_fields[1]
    y_coords = filename_fields[2]
    output["well"] = well

    # Split the well_id into row and column after first letter
    row, col = well[0], well[1:]
    x_micrometer = (
        x_coords.split("-")[1] if "-" in x_coords else x_coords.split("+")[1]
    )  # split at - or +
    y_micrometer = (
        y_coords.split("-")[1] if "-" in y_coords else y_coords.split("+")[1]
    )  # split at - or +
    output["row"] = row
    output["col"] = col
    output["x_coords"] = x_micrometer
    output["y_coords"] = y_micrometer

    return output


def extract_cellvoyager_metadata(
    metadata: pd.DataFrame, pixel_sizes_zyx_dict: dict, h5_file: str
) -> tuple[int, Point, Point, OriginDict]:
    """Extract metadata (coords, ROI)

    from Cellvoyager metadata DataFrame from h5_file name.
    """
    agg_meta = metadata.groupby(["well_id", "FieldIndex"]).agg(list)

    h5_filename = Path(h5_file).stem
    well_id, x, y = h5_filename.split("_")
    x_micrometer_h5 = x.split("-")[1] if "-" in x else x.split("+")[1]
    y_micrometer_h5 = y.split("-")[1] if "-" in y else y.split("+")[1]

    agg_meta = agg_meta.loc[well_id].reset_index()
    metawell = agg_meta.copy()
    metawell["x_micrometer"] = metawell["x_micrometer"].apply(
        lambda x: abs(round(float(x[0])))
    )
    metawell["y_micrometer"] = metawell["y_micrometer"].apply(
        lambda y: abs(round(float(y[0])))
    )

    metawell_roi = metawell[
        (metawell["x_micrometer"] == abs(round(float(x_micrometer_h5))))
        & (metawell["y_micrometer"] == abs(round(float(y_micrometer_h5))))
    ]
    if metawell_roi.empty:
        raise ValueError(
            f"No matching metadata found for {h5_filename=} "
            "in the provided Cellvoyager metadata."
        )

    ROI = int(metawell_roi["FieldIndex"].values[0])
    agg_meta = agg_meta[agg_meta["FieldIndex"] == ROI].reset_index(drop=True)

    pixel_sizes = PixelSize(
        x=pixel_sizes_zyx_dict.get("x", 1),
        y=pixel_sizes_zyx_dict.get("y", 1),
        z=pixel_sizes_zyx_dict.get("z", 1),
    )

    # Always extract the first element from list-valued columns
    assert all(agg_meta.x_micrometer[0] == x for x in agg_meta.x_micrometer)
    assert all(agg_meta.y_micrometer[0] == y for y in agg_meta.y_micrometer)
    pos_x = int(round(agg_meta.x_micrometer[0][0] / pixel_sizes.x))
    pos_y = int(round(agg_meta.y_micrometer[0][0] / pixel_sizes.y))
    size_x = agg_meta.x_pixel[0][0]
    size_y = agg_meta.y_pixel[0][0]

    min_z = min(agg_meta.z_micrometer[0])
    top_left = Point(
        x=pos_x,
        y=pos_y,
        z=min_z,
    )
    max_z = max(agg_meta.z_micrometer[0])
    bottom_right = Point(
        x=pos_x + size_x,
        y=pos_y + size_y,
        z=(max_z + 1),
    )

    origin = OriginDict(
        x_micrometer_original=pos_x,
        y_micrometer_original=pos_y,
        z_micrometer_original=0,
    )

    return ROI, top_left, bottom_right, origin


def h5_datasets(f: h5py.File, return_names=False, dsets=None) -> list[h5py.Dataset]:
    """Recursively get all datasets from an HDF5 file."""
    if dsets is None:
        dsets = []

    for group_or_dataset_name in f.keys():
        if isinstance(f[group_or_dataset_name], h5py.Group):
            h5_datasets(
                f[group_or_dataset_name], return_names=return_names, dsets=dsets
            )
        elif isinstance(f[group_or_dataset_name], h5py.Dataset):
            if return_names:
                dsets.append(f[group_or_dataset_name].name)
            else:
                dsets.append(f[group_or_dataset_name])
    return dsets


def h5_select(
    f: h5py.File,
    attrs_select: Optional[dict[str, str | int | tuple[str | int, ...]]] = None,
    not_attrs_select: Optional[dict[str, str | int | tuple[str | int, ...]]] = None,
    return_names: bool = False,
) -> h5py.Dataset:
    """Select a dataset (lazily) from an HDF5 file based on attributes."""
    dsets: list[h5py.Dataset] = []
    for dset in h5_datasets(f):
        check: list[bool] = []
        if attrs_select:
            for a in attrs_select:
                if isinstance(attrs_select[a], (tuple | list)):
                    check.append(dset.attrs.get(a) in attrs_select[a])
                else:
                    check.append(dset.attrs.get(a) == attrs_select[a])

        uncheck: list[bool] = []
        if not_attrs_select:
            for b in not_attrs_select:
                if isinstance(not_attrs_select[b] | (tuple | list)):
                    uncheck.append(dset.attrs.get(b) in not_attrs_select[b])
                else:
                    uncheck.append(dset.attrs.get(b) == not_attrs_select[b])

        if all(check) and not any(uncheck):
            if return_names:
                dsets.append(dset.name)
            else:
                dsets.append(dset)
    if len(dsets) > 1:
        logger.warning(
            "Found multiple datasets matching the selection criteria "
            f"for attributes {attrs_select} and "
            f"not attributes {not_attrs_select}. Returning the first one."
        )
    return dsets[0] if dsets else None


def h5_load(
    input_path: str,
    channel: ConverterOmeroChannel,
    level: int,
    cycle: int,
    img_type: str,
    h5_handle: Optional[h5py.File] = None,
):
    """Load a dataset from an HDF5 file based on metadata."""
    if h5_handle is not None:
        f = h5_handle
    else:
        f = h5py.File(input_path, "r")
    dset = h5_select(
        f=f,
        attrs_select={
            "img_type": img_type,
            "cycle": cycle,
            "stain": channel.label,
            "level": level,
        },
    )
    if dset is None:
        logger.warning(
            f"Dataset not found for channel {channel.label}, "
            f"wavelength {channel.wavelength_id}, cycle {cycle}, "
            f"level {level}, img_type {img_type}."
        )
        f.close()
        return None, []

    scale = dset.attrs["element_size_um"]
    # Load lazily using Dask
    arr = da.from_array(dset)
    return arr, scale, f  # Return the file handle to close it later


def find_shape(
    bottom_right: list[Point], dask_imgs: list[da.Array]
) -> tuple[int, int, int, int]:
    """Find the shape of the image."""
    max_r_x = max(bot_r.x for bot_r in bottom_right)
    max_r_y = max(bot_r.y for bot_r in bottom_right)

    shape_x = int(max_r_x)
    shape_y = int(max_r_y)

    shape_c, shape_z, *_ = dask_imgs[0].shape
    return shape_c, shape_z, shape_y, shape_x


def find_chunk_shape(
    dask_imgs: list[da.Array],
    max_xy_chunk: int = 4096,
    z_chunk: int = 1,
    c_chunk: int = 1,
) -> tuple[int, int, int, int]:
    """Find the chunk shape of the image."""
    shape_c, shape_z, shape_y, shape_x = dask_imgs[0].shape
    chunk_y = min(shape_y, max_xy_chunk)
    chunk_x = min(shape_x, max_xy_chunk)
    chunk_z = min(shape_z, z_chunk)
    chunk_c = min(shape_c, c_chunk)
    return chunk_c, chunk_z, chunk_y, chunk_x


def find_dtype(dask_imgs: list[da.Array]) -> str:
    """Find the dtype of the image."""
    return dask_imgs[0].dtype
