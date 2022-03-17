# distutils: language=c++
# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.stdint cimport int64_t


cdef extern from 'stdlib.h' nogil:
    double fabs(double value)
    
    
cdef extern from 'stdlib.h' nogil:
    int abs(int value)    


cdef extern from 'numpy/npy_math.h' nogil:
    bint npy_isnan(double x)


# nogil vector
cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        void push_back(T&) nogil
        size_t size() nogil
        T& operator[](size_t) nogil
        void clear() nogil


cdef inline double _euclidean_distance(double xloc, double yloc, double xh, double yh) nogil:
    return ((xloc - xh)**2 + (yloc - yh)**2)**0.5


cdef inline double _get_max(double v1, double v2) nogil:
    return v1 if v1 >= v2 else v2


cdef inline double _clip_high(double v, double high) nogil:
    return high if v > high else v


cdef inline double _clip(double v, double low, double high) nogil:
    return low if v < low else _clip_high(v, high)


cdef inline double _scale_min_max(double xv, double mni, double mxi, double mno, double mxo) nogil:
    return (((mxo - mno) * (xv - mni)) / (mxi - mni)) + mno


cdef double _get_mean_3d(
    double[:, :, ::1] data,
    unsigned int nbands,
    unsigned int ridx,
    unsigned int cidx
) nogil:
    """Returns the band-wise mean
    """
    cdef:
        Py_ssize_t n
        double data_mean = 0.0
        double data_val

    for n in range(0, nbands):
        data_val = data[n, ridx, cidx]
        data_mean += data_val

    return data_mean / <double>nbands


cdef double _get_max_3d(
    double[:, :, ::1] data,
    unsigned int nbands,
    unsigned int ridx,
    unsigned int cidx
) nogil:
    """Returns the band-wise maximum
    """
    cdef:
        Py_ssize_t n
        double data_max = -1e9
        double data_val

    for n in range(0, nbands):
        data_val = data[n, ridx, cidx]
        data_max = _get_max(data_val, data_max)

    return data_max


cdef double _get_max_4d(
    double[:, :, :, ::1] data,
    unsigned int ntime,
    unsigned int nbands,
    unsigned int ridx,
    unsigned int cidx
) nogil:
    """Returns the time- and band-wise maximum
    """
    cdef:
        Py_ssize_t m, n
        double data_max = -1e9
        double data_val

    for m in range(0, ntime):
        for n in range(0, nbands):
            data_val = data[m, n, ridx, cidx]
            data_max = _get_max(data_val, data_max)

    return data_max


# cdef double _get_max_4df(double[:, :, :, ::1] data,
#                          unsigned int ntime,
#                          unsigned int nbands,
#                          unsigned int nr,
#                          unsigned int nc) nogil:
#
#     cdef:
#         Py_ssize_t m, n, ridx, cidx
#         double data_max = -1e9
#         double data_val
#
#     for m in range(0, ntime):
#         for n in range(0, nbands):
#             for ridx in range(0, nr):
#                 for cidx in range(0, nc):
#
#                     data_val = data[m, n, ridx, cidx]
#
#                     data_max = _get_max(data_val, data_max)
#
#     return data_max


cdef double _determinant_transform(vector[double] t) nogil:
    """The determinant of the transform matrix.
    This value is equal to the area scaling factor when the
    transform is applied to a shape.

    Reference:
        https://github.com/sgillies/affine/blob/master/affine/__init__.py
    """
    cdef:
        double sa, sb, sc, sd, se, sf

    sa, sb, sc, sd, se, sf = t[0], t[1], t[2], t[3], t[4], t[5]

    return sa * se - sb * sd


cdef vector[double] _invert_transform(vector[double] t) nogil:
    """Returns the inverse transform

    Reference:
        https://github.com/sgillies/affine/blob/master/affine/__init__.py
    """
    cdef:
        vector[double] t_
        double idet
        double sa, sb, sc, sd, se, sf
        double ra, rb, rd, re

    idet = 1.0 / _determinant_transform(t)
    sa, sb, sc, sd, se, sf = t[0], t[1], t[2], t[3], t[4], t[5]
    ra = se * idet
    rb = -sb * idet
    rd = -sd * idet
    re = sa * idet

    t_.push_back(ra)
    t_.push_back(rb)
    t_.push_back(-sc * ra - sf * rb)
    t_.push_back(rd)
    t_.push_back(re)
    t_.push_back(-sc * rd - sf * re)

    return t_


cdef void _transform_coords_to_indices(
    vector[double] t,
    double vx,
    double vy,
    int64_t[::1] out_indices__
) nogil:
    """Transforms coordinates to indices
    
    Reference:
        https://github.com/sgillies/affine/blob/master/affine/__init__.py
    """
    cdef:
        double sa, sb, sc, sd, se, sf

    sa, sb, sc, sd, se, sf = t[0], t[1], t[2], t[3], t[4], t[5]

    out_indices__[0] = <int>(vx * sa + vy * sb + sc)
    out_indices__[1] = <int>(vx * sd + vy * se + sf)


cdef void _coarse_transformer(
    Py_ssize_t i,
    Py_ssize_t j,
    unsigned int kh,
    vector[double] hr_transform_,
    vector[double] cr_transform_,
    int64_t[::1] out_indices_
) nogil:
    """Transforms coordinates to indices for a coarse-to-high resolution transformation

    Args:
        i (int): The row index position for the high-resolution grid.
        j (int): The column index position for the high-resolution grid.
        kh (int): The center pixel offset for the high-resolution grid.
        hr_transform (list): The high-resolution affine transform.
        cr_transform (list): The coarse-resolution affine transform.
    """
    cdef:
        double x, y
        Py_ssize_t row_index, col_index

    # Coordinates of the high-resolution center pixel
    x = hr_transform_[2] + ((j+kh) * fabs(hr_transform_[0]))
    y = hr_transform_[5] - ((i+kh) * fabs(hr_transform_[4]))

    # Invert the coarse resolution affine transform and
    # get the indices at the x,y coordinates.
    _transform_coords_to_indices(_invert_transform(cr_transform_), x, y, out_indices_)


cdef class SingleSensorNetwork(object):
    """A network class for a single sensor
    """
    cdef:
        int64_t[:, ::1] grid
        vector[int] edge_indices_a
        vector[int] edge_indices_b
        vector[double] edge_attrs_diffs, edge_attrs_dists
        vector[double] xpos, ypos
        unsigned int nbands, nrows, ncols
        double[:, :, ::1] varray
        int k_, kh
        double cell_size_
        double max_dist, max_scaled, eps

    def __init__(
        self,
        double[:, :, ::1] value_array,
        int k=3,
        float cell_size=30.0
    ):
        self.nbands = value_array.shape[0]
        self.nrows = value_array.shape[1]
        self.ncols = value_array.shape[2]
        self.k_ = k
        self.kh = <int>(self.k_ * 0.5)
        self.cell_size_ = cell_size
        self.varray = value_array

        self.max_dist = _euclidean_distance(0.0, 0.0, <double>self.kh, <double>self.kh)
        self.max_scaled = 1.0 - (_euclidean_distance(<double>self.kh, <double>self.kh-1, <double>self.kh, <double>self.kh) / self.max_dist)
        self.eps = 1e-6

        self.grid = np.arange(0, self.nrows*self.ncols).reshape(self.nrows, self.ncols).astype('int64')

    def create_network(self):
        self._create_network()

        return self.edge_indices_a, self.edge_indices_b, self.edge_attrs_diffs, self.edge_attrs_dists, self.xpos, self.ypos

    cdef void _create_network(self) nogil:
        cdef:
            Py_ssize_t i, j, m, n
            bint do_connect
            unsigned int column_end = self.ncols - self.k_

        for i in range(0, self.nrows-self.k_):
            for j in range(0, self.ncols-self.k_):
                # Connect to center node
                for m in range(0, self.k_):
                    for n in range(0, self.k_):
                        if m+1 < self.k_:
                            do_connect = True
                            if (i > 0) and (j == 0):
                                if m < self.kh:
                                    do_connect = False

                            elif j > 0:
                                # Only the second column half of the window needs updated
                                if n <= self.kh:
                                    do_connect = False

                            if do_connect:
                                # Vertical connection
                                self._connect_window(m, n, m+1, n, i, j)

                        if n+1 < self.k_:
                            do_connect = True
                            if (i > 0) and (j == 0):
                                if m <= self.kh:
                                    do_connect = False

                            elif j > 0:
                                if n < self.kh:
                                    do_connect = False

                            if do_connect:
                                # Horizontal connection
                                self._connect_window(m, n, m, n+1, i, j)

                        if (j == 0) and (m == 0) and (n == self.kh):
                            self._connect_window(m, n, self.kh, 0, i, j)

                        if (j == column_end) and (m == 0) and (n == self.kh):
                            self._connect_window(m, n, self.kh, self.k_-1, i, j)

                        # Avoid already connected direct neighbors
                        # o - x - o
                        # | \ | / |
                        # x - O - x
                        # | / | \ |
                        # o - x - o
                        if abs(m - self.kh) + abs(n - self.kh) > self.kh:
                            # Diagonal edges
                            self._connect_window(m, n, self.kh, self.kh, i, j)

    cdef void _connect_window(
        self,
        Py_ssize_t isource,
        Py_ssize_t jsource,
        Py_ssize_t itarg,
        Py_ssize_t jtarg,
        Py_ssize_t idx,
        Py_ssize_t jdx,
        bint directed=False
    ) nogil:
        """
        Args:
            isource (int): The source window row index.
            jsource (int): The source window column index.
            itarg (int): The target window row index.
            jtarg (int): The target window column index.
            idx (int): The array row index.
            jdx (int): The array column index.
            max_dist (float): The maximum window distance from the center.
            eps (float): An offset value to avoid zero weights.
        """
        cdef:
            Py_ssize_t b
            double w, val_diff

        # COO format:
        # [[sources, ...]
        #  [targets, ...]]

        # Center to link
        self.edge_indices_a.push_back(self.grid[idx+isource, jdx+jsource])
        self.edge_indices_b.push_back(self.grid[idx+itarg, jdx+jtarg])

        if not directed:
            # Link to center
            self.edge_indices_a.push_back(self.grid[idx+itarg, jdx+jtarg])
            self.edge_indices_b.push_back(self.grid[idx+isource, jdx+jsource])

        w = 1.0 - (_euclidean_distance(<double>jsource, <double>isource, <double>jtarg, <double>itarg) / self.max_dist)

        w = _scale_min_max(w, 0.0, self.max_scaled, 0.75, 1.0)
        w = _clip(w, 0.75, 1.0)

        if npy_isnan(w):
            w = self.eps

        val_diff = 0.0
        for b in range(0, self.nbands):
            val_diff += self.varray[b, idx+isource, jdx+jsource] - self.varray[b, idx+itarg, jdx+jtarg]

        val_diff /= <double>self.nbands

        val_diff = _clip(fabs(val_diff), 0.0, 1.0)
        val_diff = _scale_min_max(val_diff, 0.0, 1.0, self.eps, 1.0)
        val_diff = _clip(val_diff, self.eps, 1.0)

        if npy_isnan(val_diff):
            val_diff = self.eps

        # Edge attributes
        self.edge_attrs_diffs.push_back(val_diff)
        self.edge_attrs_dists.push_back(w)

        # x, y coordinates
        self.xpos.push_back((jdx+jtarg)*self.cell_size_)
        self.ypos.push_back((idx+itarg)*self.cell_size_)

        if not directed:
            self.edge_attrs_diffs.push_back(val_diff)
            self.edge_attrs_dists.push_back(w)
            self.xpos.push_back((jdx+jsource)*self.cell_size_)
            self.ypos.push_back((idx+isource)*self.cell_size_)

        
cdef class MultiSensorNetwork(object):
    """A class for a multi-sensor network
    """
    cdef:
        unsigned int ntime, nbands, nrows_, ncols_
        double[:, :, :, ::1] xarray
        double[:, :, ::1] yarray
        vector[vector[double]] transforms_
        unsigned int n_transforms_
        int64_t[:, ::1] grid_
        vector[int64_t[:, ::1]] grid_c_
        vector[double[:, :, :, ::1]] grid_c_resamp_
        unsigned int k_, kh
        double nodata_
        double coarse_window_res_limit_
        double max_edist_hres_
        bint add_coarse_nodes_

        vector[int] edge_indices_a
        vector[int] edge_indices_b
        vector[double] edge_attrs
    """Creates graph edges and edge attributes
    
    Args:
        xdata (4d array): [time x bands x rows x columns]
        ydata (3d array): [band x rows x columns]
        nrows (int)
        ncols (int)
        transforms (list)
        direct_to_center (bool): Whether to direct edges connected to the center pixel (i.e., in one direction).
        add_coarse_nodes (bool): Whether to add coarse resolution data as separate nodes.
        k (int): The local window size.
        nodata (float | int)
        coarse_window_res_limit (float | int)
    """

    def __init__(
        self,
        double[:, :, :, ::1] xdata,
        double[:, :, ::1] ydata,
        vector[vector[double]] transforms,
        unsigned int n_transforms,
        int64_t[:, ::1] grid,
        vector[int64_t[:, ::1]] grid_c,
        vector[double[:, :, :, ::1]] grid_c_resamp,
        bint direct_to_center=False,
        bint add_coarse_nodes=False,
        unsigned int k=7,
        double nodata=0.0,
        double coarse_window_res_limit=30.0,
        double max_edist_hres=1.0
    ):
        self.xarray = xdata
        self.yarray = ydata
        self.transforms_ = transforms

        self.n_transforms_ = n_transforms

        self.ntime = self.xarray.shape[0]
        self.nbands = self.xarray.shape[1]
        self.nrows_ = self.xarray.shape[2]
        self.ncols_ = self.xarray.shape[3]

        # 1:1 grid for high-res y and high-res X variables
        self.grid_ = grid
        self.grid_c_ = grid_c
        self.grid_c_resamp_ = grid_c_resamp

        self.add_coarse_nodes_ = add_coarse_nodes
        self.k_ = k
        self.kh = <int>(self.k_ / 2.0)
        self.nodata_ = nodata
        self.coarse_window_res_limit_ = coarse_window_res_limit
        self.max_edist_hres_ = max_edist_hres

    def create_network(self):
        cdef:
            Py_ssize_t i, j
            int64_t[::1] out_indices = np.zeros(2, dtype='int64')
            double[:, :, :, ::1] xarray_ = self.xarray
            double[:, :, ::1] yarray_ = self.yarray
            int64_t[:, ::1] grid_ = self.grid_

        with nogil:
            # Create node edges and edge weights
            for i in range(0, self.nrows_-self.k_):
                for j in range(0, self.ncols_-self.k_):
                    # Local window iteration for direct neighbors
                    self.create_hr_nodes(i, j, xarray_, yarray_, grid_)
                    if self.add_coarse_nodes_:
                        self.create_coarse_undirected_isolated(i, j, self.kh, out_indices)
                        self.create_coarse_center_edges(i, j, self.kh, out_indices, yarray_, grid_)

        return self.edge_indices_a, self.edge_indices_b, self.edge_attrs

    cdef void _connect_window(
        self,
        int64_t[:, ::1] grid_,
        double[:, :, :, ::1] xarray,
        double[:, :, ::1] yarray,
        Py_ssize_t targ_i,
        Py_ssize_t targ_j,
        Py_ssize_t idx,
        Py_ssize_t jdx,
        Py_ssize_t source_i,
        Py_ssize_t source_j,
        bint center_weights,
        double weight_gain
    ) nogil:
        """
        Args:
            grid_ (2d array): The grid indices.
            xarray (4d array)
            yarray (3d array)
            targ_i (int): The target window row index.
            targ_j (int): The target window column index.
            idx (int): The array row index.
            jdx (int): The array column index.
            source_i (int): The source window row index.
            source_j (int): The source window column index.
        """
        cdef:
            double edge_weight
            double edist, spdist
            double mean_off, mean_center
        
        # COO format:
        # [[sources, ...]
        #  [targets, ...]]     
        
        # Source -> target
        self.edge_indices_a.push_back(grid_[idx+source_i, jdx+source_j])
        self.edge_indices_b.push_back(grid_[idx+targ_i, jdx+targ_j])

        # Target -> source
        self.edge_indices_a.push_back(grid_[idx+targ_i, jdx+targ_j])
        self.edge_indices_b.push_back(grid_[idx+source_i, jdx+source_j])    

        # Both arrays must have data in the neighbors
        if (_get_max_4d(xarray, self.ntime, self.nbands, idx+source_i, jdx+source_j) != self.nodata_) and \
            (_get_max_4d(xarray, self.ntime, self.nbands, idx+targ_i, jdx+targ_j) != self.nodata_) and \
            (_get_max_3d(yarray, self.nbands, idx+source_i, jdx+source_j) != self.nodata_) and \
            (_get_max_3d(yarray, self.nbands, idx+targ_i, jdx+targ_j) != self.nodata_):

            if center_weights:
                # Inverse euclidean distance
                edist = 1.0 - ((_euclidean_distance(<double>self.kh, <double>self.kh, <double>source_i, <double>source_j) * self.transforms_[0][0]) / self.max_edist_hres_)

                # Inverse spectral difference
                mean_off = _get_mean_3d(yarray, self.nbands, idx+source_i, jdx+source_j)
                mean_center = _get_mean_3d(yarray, self.nbands, idx+targ_i, jdx+targ_j)

                spdist = 1.0 - fabs(mean_off - mean_center)

                # max(edist, spdist) x 10
                edge_weight = _get_max(edist, spdist) * weight_gain                
            
            else:
                if (targ_i == self.kh) and (targ_j == self.kh):
                    edge_weight = 1.0
                else:
                    edge_weight = 0.5

            self.edge_attrs.push_back(edge_weight)
            self.edge_attrs.push_back(edge_weight)

        else:
            self.edge_attrs.push_back(0.0)
            self.edge_attrs.push_back(0.0)    
    
    cdef void create_hr_nodes(
        self,
        Py_ssize_t i,
        Py_ssize_t j,
        double[:, :, :, ::1] xarray,
        double[:, :, ::1] yarray,
        int64_t[:, ::1] grid_,
        double hr_weight=5.0
    ) nogil:
        """Creates high-resolution nodes and edges
        """
        cdef:
            Py_ssize_t m, n
            bint do_connect
            
        # Connect to center node
        for m in range(0, self.k_):
            for n in range(0, self.k_):
                if m+1 < self.k_:
                    do_connect = True
                    if (i > 0) and (j == 0):
                        if m < self.kh:
                            do_connect = False
                    
                    elif j > 0:
                        # Only the second column half of the window needs updated
                        if n <= self.kh:
                            do_connect = False
                            
                    if do_connect:
                        # Vertical connection
                        # (grid, targ_i, targ_j, i, j, source_i, source_j)
                        self._connect_window(grid_, xarray, yarray, m+1, n, i, j, m, n, False, hr_weight)

                if n+1 < self.k_:
                    do_connect = True
                    if (i > 0) and (j == 0):
                        if m <= self.kh:
                            do_connect = False
                    
                    elif j > 0:
                        if n < self.kh:
                            do_connect = False
                            
                    if do_connect:
                        # Horizontal connection
                        self._connect_window(grid_, xarray, yarray, m, n+1, i, j, m, n, False, hr_weight)

                # Avoid already connected direct neighbors
                # o - x - o
                # | \ | / |
                # x - O - x
                # | / | \ |
                # o - x - o
                if abs(m - self.kh) + abs(n - self.kh) <= 1:
                    continue
                    
                # Diagonal edges
                self._connect_window(grid_, xarray, yarray, self.kh, self.kh, i, j, m, n, True, hr_weight)

    cdef void create_coarse_undirected_isolated(
        self,
        Py_ssize_t i,
        Py_ssize_t j,
        unsigned int kh,
        int64_t[::1] out_indices
    ) nogil:
        """Creates undirected, isolated (from the center) edges for coarse grids
        """
        cdef:
            vector[double] hr_transform, cr_transform
            Py_ssize_t pidx
            int64_t[:, ::1] coarse_grid
            double[:, :, :, ::1] coarse_xarray
            unsigned int ntime_, nbands_, nr, nc
            unsigned int row_off, col_off
            unsigned int row_off_nbr, col_off_nbr

        # Static 3x3 window for coarse grids
        hr_transform = self.transforms_[0]

        for pidx in range(0, self.n_transforms_-1):
            cr_transform = self.transforms_[pidx+1]
            # Do not connect extremely coarse grids
            if fabs(cr_transform[0]) > self.coarse_window_res_limit_:
                continue

            coarse_grid = self.grid_c_[pidx]
            coarse_xarray = self.grid_c_resamp_[pidx]

            ntime_ = coarse_xarray.shape[0]
            nbands_ = coarse_xarray.shape[1]
            nr = coarse_xarray.shape[2]
            nc = coarse_xarray.shape[3]

            # Get the row/column indices of the coarse resolution
            # that intersect the high-resolution.
            _coarse_transformer(
                i,
                j,
                kh,
                hr_transform,
                cr_transform,
                out_indices
            )

            col_off = out_indices[0]
            row_off = out_indices[1]

            if row_off > nr - 1:
                row_off = nr - 1

            if col_off > nc - 1:
                col_off = nc - 1

            row_off_nbr = row_off + 1
            col_off_nbr = col_off + 1

            if col_off < nc - 1:
                # Edge 1
                # n1 --> n2
                self.edge_indices_a.push_back(coarse_grid[row_off, col_off])
                self.edge_indices_b.push_back(coarse_grid[row_off, col_off_nbr])

                # Edge 2
                # n1 <-- n2
                self.edge_indices_a.push_back(coarse_grid[row_off, col_off_nbr])
                self.edge_indices_b.push_back(coarse_grid[row_off, col_off])

                # Both arrays must have data in the neighbor and at the center
                if (_get_max_4d(coarse_xarray, ntime_, nbands_, row_off, col_off) != self.nodata_) and \
                                (_get_max_4d(coarse_xarray, ntime_, nbands_, row_off, col_off_nbr) != self.nodata_):

                    self.edge_attrs.push_back(0.1)
                    self.edge_attrs.push_back(0.1)

                else:
                    self.edge_attrs.push_back(0.0)
                    self.edge_attrs.push_back(0.0)

            if row_off < nr - 1:
                # Edge 1
                # n1
                # ^
                # |
                # n2
                self.edge_indices_a.push_back(coarse_grid[row_off, col_off])
                self.edge_indices_b.push_back(coarse_grid[row_off_nbr, col_off])

                # Edge 2
                # n1
                # |
                # v
                # n2
                self.edge_indices_a.push_back(coarse_grid[row_off_nbr, col_off])
                self.edge_indices_b.push_back(coarse_grid[row_off, col_off])

                # Both arrays must have data in the neighbor and at the center
                if (_get_max_4d(coarse_xarray, ntime_, nbands_, row_off, col_off) != self.nodata_) and \
                                (_get_max_4d(coarse_xarray, ntime_, nbands_, row_off_nbr, col_off) != self.nodata_):

                    self.edge_attrs.push_back(0.1)
                    self.edge_attrs.push_back(0.1)

                else:
                    self.edge_attrs.push_back(0.0)
                    self.edge_attrs.push_back(0.0)

    cdef void create_coarse_center_edges(
        self,
        Py_ssize_t i,
        Py_ssize_t j,
        unsigned int kh,
        int64_t[::1] out_indices,
        double[:, :, ::1] yarray,
        int64_t[:, ::1] grid_
    ) nogil:
        """Creates edges from the coarse resolution to high-resolution center
        """
        cdef:
            Py_ssize_t ii, jj
            unsigned int kh_
            Py_ssize_t row_center, col_center
            vector[double] hr_transform, cr_transform
            Py_ssize_t pidx
            int64_t[:, ::1] coarse_grid
            int64_t[:, ::1] prev_grid
            double[:, :, :, ::1] coarse_xarray
            unsigned int nr, nc
            unsigned int row_index, col_index
            unsigned int row_off_nbr, col_off_nbr
            double edge_weight, weight_step, baseline_weight

        # The first grid edge weights
        edge_weight = 1.0
        weight_step = -0.5
        baseline_weight = 0.1

        for pidx in range(0, self.n_transforms_-1):
            # Get the transform vectors
            hr_transform = self.transforms_[pidx]
            cr_transform = self.transforms_[pidx+1]

            # Get the grid of the previous resolution
            if pidx == 0:
                prev_grid = grid_
                ii = i
                jj = j
                kh_ = kh
                row_center = i + kh_
                col_center = j + kh_

            else:
                prev_grid = self.grid_c_[pidx-1]
                ii = row_index
                jj = col_index
                kh_ = 0
                row_center = row_index
                col_center = col_index

            if row_center >= prev_grid.shape[0] - 1:
                row_center = prev_grid.shape[0] - 1

            if col_center >= prev_grid.shape[1] - 1:
                col_center = prev_grid.shape[1] - 1

            # Get the current coarse(r) resolution grid
            coarse_grid = self.grid_c_[pidx]

            # Get the current resampled coarse resolution data
            coarse_xarray = self.grid_c_resamp_[pidx]

            ntime_ = coarse_xarray.shape[0]
            nbands_ = coarse_xarray.shape[1]
            nr = coarse_xarray.shape[2]
            nc = coarse_xarray.shape[3]

            # Get the row/column indices of the coarse resolution
            # that intersects the high-resolution.
            _coarse_transformer(
                ii,
                jj,
                kh_,
                hr_transform,
                cr_transform,
                out_indices
            )

            # Row/column indices for the coarse(r) resolution center pixel
            col_index = out_indices[0]
            row_index = out_indices[1]

            if row_index > nr - 1:
                row_index = nr - 1

            if col_index > nc - 1:
                col_index = nc - 1

            # Coarse-res edge links to the center_y
            self.edge_indices_a.push_back(coarse_grid[row_index, col_index])
            self.edge_indices_b.push_back(prev_grid[row_center, col_center])

            self.edge_indices_a.push_back(prev_grid[row_center, col_center])
            self.edge_indices_b.push_back(coarse_grid[row_index, col_index])

            # Both arrays must have data in the neighbor and at the center
            if (_get_max_4d(coarse_xarray, ntime_, nbands_, row_index, col_index) != self.nodata_) and \
                (_get_max_3d(yarray, self.nbands, row_center, col_center) != self.nodata_):

                self.edge_attrs.push_back(edge_weight)
                self.edge_attrs.push_back(edge_weight)

            else:
                self.edge_attrs.push_back(0.0)
                self.edge_attrs.push_back(0.0)

            edge_weight += weight_step

            if edge_weight < baseline_weight:
                edge_weight = baseline_weight        
