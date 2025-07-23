### Purpose
- **Computes image-based registration** transformations for acquisitions in **HCS** OME-Zarr datasets using the warpfield library.
- Needs warpfield registration recipe to configure the registration, otherwise applying default registration.
- Calculates registration transforms e.g. per FOV by providing FOV_ROI_table.
- Can handle cases where there are more than one embryo / organoid in a FOV if each ROI e.g. embryo / organoid is masked by a linked label (e.g. calculated by`scMultiplex Calculate Object Linking`) and corresponding masking_roi_table.
- Calculates transformations for **specified regions of interest (ROIs)** and stores the results in a registration subfolder per OME-Zarr image.
- Typically used as the first task in a workflow, followed by `Apply Registration (warpfield)`.

### Limitations
- Supports only HCS OME-Zarr datasets, leveraging their acquisition metadata and well-based image grouping.
