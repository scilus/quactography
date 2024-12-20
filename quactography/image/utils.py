import numpy as np

AXES_MAPPING = {
    'sagittal': 0,
    'coronal': 1,
    'axial': 2
}


def slice_along_axis(arr, axis_name, slice_index):
    """
    Slice a 3D array along a given axis

    Parameters
    ----------
    arr : numpy array
        Array to slice.
    axis_name : str
        Name of the axis along which to slice.
    slice_index : int
        Index of the slice to take.
    Returns
    -------
    slice : numpy array
        Slice of the input array, along the specified axis
    """
    ax_id = AXES_MAPPING[axis_name.lower()]
    slice_idx = slice_index if slice_index is not None else arr.shape[ax_id]
    return np.take(arr, slice_idx, ax_id)
