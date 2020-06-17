# MoCapLib
MoCapLib: Library for Motion Capture data processing

## Description
The aim of this repository is to provide a simple Python-based library for motion capture data processing and analysis.

## Installation
MoCapLib can be installed from [PyPI](https://pypi.org/project/mocaplib/) using ```pip``` on Python>=3.7.

```bash
pip install mocaplib
```

## Usage
Currently, there is only one sub-module availabe, 'mocaplib.gapfill'.
```python
import mocaplib as mcl
import mocaplib.gapfill as gf

# numpy is required in order to provide necessary markers' coordinate values
import numpy as np

# 'fill_marker_gap_interp()' function will update the given ndarray by filling its gaps using bspline interpolation
# 'tgt_mkr_pos0': a 2D ndarray with the shape of (n, 3) of a target marker position to fill the gaps
# 'n' is the total number of frames
tgt_mkr_pos0 = np.array((n, 3), dtype=np.float32)
# 'ret0': either True or False, True if there is any frame updated, False if there is no frame updated
# 'updated_frs_mask0': a boolean ndarray to indicate which frames are updated
ret0, updated_frs_mask0 = gf.fill_marker_gap_interp(tgt_mkr_pos0)

# 'fill_marker_gap_pattern()' function will update the given ndarray by filling its gaps using a donor marker
# 'tgt_mkr_pos1': a 2D ndarray with the shape of (n, 3) of a target marker position to fill the gaps
# 'n' is the total number of frames
tgt_mkr_pos1 = np.array((n, 3), dtype=np.float32)
# a 2D ndarray with the shape of (n, 3) of a donor marker position
# 'n' is the total number of frames
dnr_mkr_pos = np.array((n, 3), dtype=np.float32)
# 'ret1': either True or False, True if there is any frame updated, False if there is no frame updated
# 'updated_frs_mask1': a boolean ndarray to indicate which frames are updated
ret1, updated_frs_mask1 = gf.fill_marker_gap_pattern(tgt_mkr_pos1, dnr_mkr_pos)

# 'fill_marker_gap_rbt()' function will update the given ndarray by filling its gaps using a cluster of 3 markers
# 'tgt_mkr_pos2': a 2D ndarray with the shape of (n, 3) of a target marker position to fill the gaps
# 'n' is the total number of frames
tgt_mkr_pos2 = np.array((n, 3), dtype=np.float32)
# 'cl_mkr_pos': a 3D ndarray with the shape of (m, n, 3) of the cluster markers
# 'm' (at least 3) is the number of markers, and 'n' is the total number of frames
cl_mkr_pos = np.array((m, n, 3), dtype=np.float32)
# 'ret2': either True or False, True if there is any frame updated, False if there is no frame updated
# 'updated_frs_mask2': a boolean ndarray to indicate which frames are updated
ret2, updated_frs_mask2 = gf.fill_marker_gap_rbt(tgt_mkr_pos2, cl_mkr_pos)
```
## Dependencies
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)

## License
[MIT](https://choosealicense.com/licenses/mit/)
