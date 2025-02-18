### Purpose
- **Computes image-based registration** transformations for acquisitions in **HCS** OME-Zarr datasets using the elastix library.
- Needs Elastix profiles to configure the registration.
- Can handle cases where there are more than one embryo / organoid in a FOV.
- Processes images grouped by well, under the assumption that each ROI e.g. embryo / organoid is masked by a linked label (e.g. calculated by
    `scMultiplex Calculate Object Linking`)  and corresponding masking_roi_table.
- Calculates transformations for **specified regions of interest (ROIs)** and stores the results in a registration subfolder per OME-Zarr image.
- Typically used as the first task in a workflow, followed by `Apply Registration ROI (elastix)`.

### Limitations
- Supports only HCS OME-Zarr datasets, leveraging their acquisition metadata and well-based image grouping.
- Assumes each well contains a single image per acquisition.
