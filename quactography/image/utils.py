import numpy as np

AXES_MAPPING = {
    'sagittal': 0,
    'coronal': 1,
    'axial': 2
}


def slice_along_axis(arr, axis_name, slice_index):
    ax_id = AXES_MAPPING[axis_name.lower()]
    slice_idx = slice_index if slice_index is not None else arr.shape[ax_id] // 2
    return np.take(arr, slice_idx, ax_id)
