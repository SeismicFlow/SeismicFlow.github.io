# seismic_processing.pyx
# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
cimport numpy as np
import numpy as np
from libc.string cimport memcpy

ctypedef float float32_t
ctypedef double float64_t
ctypedef int int32_t

# Optimized float32 implementation
cdef void c_populate_seismic_cube_float32(
    const float32_t* amplitudes,
    const int32_t* inline_values,
    const int32_t* crossline_values,
    const unordered_map[int32_t, int]& inline_map,
    const unordered_map[int32_t, int]& crossline_map,
    float32_t* output_cube,
    const int n_rows,
    const int inline_max,
    const int crossline_max,
    const int time_intervals) noexcept nogil:

    cdef:
        int i
        int inline_idx, crossline_idx
        int32_t inline_val, crossline_val
        int flat_idx
        int stride_inline = crossline_max * time_intervals
        int stride_crossline = time_intervals
        size_t copy_size = time_intervals * sizeof(float32_t)
        const float32_t* amp_ptr
        float32_t* out_ptr
        vector[int] flat_indices
    
    # Pre-calculate all flat indices
    flat_indices.reserve(n_rows)
    for i in range(n_rows):
        inline_val = inline_values[i]
        crossline_val = crossline_values[i]
        
        inline_idx = inline_map.at(inline_val)
        crossline_idx = crossline_map.at(crossline_val)
        
        flat_indices.push_back(inline_idx * stride_inline + crossline_idx * stride_crossline)
    
    # Bulk copy data using pre-calculated indices
    for i in range(n_rows):
        amp_ptr = amplitudes + i * time_intervals
        out_ptr = output_cube + flat_indices[i]
        memcpy(out_ptr, amp_ptr, copy_size)

# Optimized float64 implementation
cdef void c_populate_seismic_cube_float64(
    const float64_t* amplitudes,
    const int32_t* inline_values,
    const int32_t* crossline_values,
    const unordered_map[int32_t, int]& inline_map,
    const unordered_map[int32_t, int]& crossline_map,
    float64_t* output_cube,
    const int n_rows,
    const int inline_max,
    const int crossline_max,
    const int time_intervals) noexcept nogil:

    cdef:
        int i
        int inline_idx, crossline_idx
        int32_t inline_val, crossline_val
        int flat_idx
        int stride_inline = crossline_max * time_intervals
        int stride_crossline = time_intervals
        size_t copy_size = time_intervals * sizeof(float64_t)
        const float64_t* amp_ptr
        float64_t* out_ptr
        vector[int] flat_indices
    
    flat_indices.reserve(n_rows)
    for i in range(n_rows):
        inline_val = inline_values[i]
        crossline_val = crossline_values[i]
        
        inline_idx = inline_map.at(inline_val)
        crossline_idx = crossline_map.at(crossline_val)
        
        flat_indices.push_back(inline_idx * stride_inline + crossline_idx * stride_crossline)
    
    for i in range(n_rows):
        amp_ptr = amplitudes + i * time_intervals
        out_ptr = output_cube + flat_indices[i]
        memcpy(out_ptr, amp_ptr, copy_size)

# Optimized int32 implementation
cdef void c_populate_seismic_cube_int32(
    const int32_t* amplitudes,
    const int32_t* inline_values,
    const int32_t* crossline_values,
    const unordered_map[int32_t, int]& inline_map,
    const unordered_map[int32_t, int]& crossline_map,
    int32_t* output_cube,
    const int n_rows,
    const int inline_max,
    const int crossline_max,
    const int time_intervals) noexcept nogil:

    cdef:
        int i
        int inline_idx, crossline_idx
        int32_t inline_val, crossline_val
        int flat_idx
        int stride_inline = crossline_max * time_intervals
        int stride_crossline = time_intervals
        size_t copy_size = time_intervals * sizeof(int32_t)
        const int32_t* amp_ptr
        int32_t* out_ptr
        vector[int] flat_indices
    
    flat_indices.reserve(n_rows)
    for i in range(n_rows):
        inline_val = inline_values[i]
        crossline_val = crossline_values[i]
        
        inline_idx = inline_map.at(inline_val)
        crossline_idx = crossline_map.at(crossline_val)
        
        flat_indices.push_back(inline_idx * stride_inline + crossline_idx * stride_crossline)
    
    for i in range(n_rows):
        amp_ptr = amplitudes + i * time_intervals
        out_ptr = output_cube + flat_indices[i]
        memcpy(out_ptr, amp_ptr, copy_size)

# Python wrapper (mostly unchanged)
def py_populate_seismic_cube(
    np.ndarray amplitudes not None,
    np.ndarray[int32_t, ndim=1] inline_values not None,
    np.ndarray[int32_t, ndim=1] crossline_values not None,
    dict py_inline_index,
    dict py_crossline_index,
    int inline_max,
    int crossline_max,
    int time_intervals):
    
    if not amplitudes.flags['C_CONTIGUOUS']:
        amplitudes = np.ascontiguousarray(amplitudes)
    
    cdef:
        unordered_map[int32_t, int] inline_map
        unordered_map[int32_t, int] crossline_map
        np.ndarray output_array
        np.ndarray[float32_t, ndim=2] amplitudes_f32
        np.ndarray[float64_t, ndim=2] amplitudes_f64
        np.ndarray[int32_t, ndim=2] amplitudes_i32
        np.ndarray[float32_t, ndim=3] output_f32
        np.ndarray[float64_t, ndim=3] output_f64
        np.ndarray[int32_t, ndim=3] output_i32

    # Pre-size the maps to avoid rehashing
    inline_map.reserve(len(py_inline_index))
    crossline_map.reserve(len(py_crossline_index))
    
    for k, v in py_inline_index.items():
        inline_map[k] = v
    for k, v in py_crossline_index.items():
        crossline_map[k] = v

    output_array = np.zeros((inline_max, crossline_max, time_intervals), dtype=amplitudes.dtype, order='C')

    if amplitudes.dtype == np.float32:
        amplitudes_f32 = amplitudes
        output_f32 = output_array
        c_populate_seismic_cube_float32(
            &amplitudes_f32[0,0],
            &inline_values[0],
            &crossline_values[0],
            inline_map,
            crossline_map,
            &output_f32[0,0,0],
            inline_values.shape[0],
            inline_max,
            crossline_max,
            time_intervals
        )
    elif amplitudes.dtype == np.float64:
        amplitudes_f64 = amplitudes
        output_f64 = output_array
        c_populate_seismic_cube_float64(
            &amplitudes_f64[0,0],
            &inline_values[0],
            &crossline_values[0],
            inline_map,
            crossline_map,
            &output_f64[0,0,0],
            inline_values.shape[0],
            inline_max,
            crossline_max,
            time_intervals
        )
    elif amplitudes.dtype == np.int32:
        amplitudes_i32 = amplitudes
        output_i32 = output_array
        c_populate_seismic_cube_int32(
            &amplitudes_i32[0,0],
            &inline_values[0],
            &crossline_values[0],
            inline_map,
            crossline_map,
            &output_i32[0,0,0],
            inline_values.shape[0],
            inline_max,
            crossline_max,
            time_intervals
        )
    else:
        raise TypeError(f"Unsupported amplitude dtype: {amplitudes.dtype}. "
                      "Supported types are: float32, float64, and int32")

    return output_array
