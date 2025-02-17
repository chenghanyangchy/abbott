### Purpose
- **Applies pre-calculated registration** from `Calculate Channel Registration (elastix)` task to images in an **HCS** OME-Zarr dataset, aligning all channels of an acquisition to a specified reference wavelength.
- This task is useful if there are wavelength- and sample-dependent chromatic shifts.
- Replaces the non-aligned image with the newly aligned image in the dataset if `overwrite input` is selected.
- Typically used as the second task in a workflow, following `Calculate Channel Registration (elastix)`.

### Limitations
- If `overwrite input` is selected, the non-aligned image is permanently deleted, which may impact workflows requiring access to the original images.
