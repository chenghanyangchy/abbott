### Purpose
- Upsamples segmented **labels** in 2D or 3D images to the highest image resolution.
- Useful if segmentation was performed at a lower resolution (e.g. level 1).

### Outputs
- A new **upsampled label image** with resolution matching those of the OME-Zarr images.
- Preserves the integer label values from the original segmentation.
