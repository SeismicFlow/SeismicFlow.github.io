# setupseismic_processing.py
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

# Enable Cython compiler optimizations
Options.docstrings = False  # Remove docstrings
Options.annotate = False    # Disable annotation files
Options.emit_code_comments = False  # Remove code comments

# MSVC-specific compiler and linker flags
extra_compile_args = [
    "/Ox",              # Maximum optimization
    "/GL",              # Whole program optimization
    "/GF",              # String pooling
    "/Gy",              # Function-level linking
    "/std:c++17",       # C++17 standard
    "/DNDEBUG",         # Disable debug code
    "/fp:fast",         # Fast floating-point model
    "/arch:AVX2",       # Use AVX2 instructions if available
    "/MP",              # Multi-processor compilation
    "/EHsc",            # Exception handling model
    "/W0",             # Disable warnings
]

extra_link_args = [
    "/LTCG",            # Link-time code generation
    "/OPT:REF",         # Eliminate unreferenced data
    "/OPT:ICF",         # Identical COMDAT folding
    "/MANIFEST:NO",     # No manifest
]

# Define extension
extensions = [
    Extension(
        "seismic_processing",
        ["seismic_processing.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_WITHOUT_ASSERTIONS", "1"),  # Disable assertions
            ("CYTHON_TRACE", "0"),              # Disable tracing
            ("CYTHON_PROFILE", "0"),            # Disable profiling
        ],
    )
]

# Setup configuration
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'overflowcheck': False,
            'embedsignature': False,
            'cdivision': True,
            'binding': True,
            'linetrace': False,
            'profile': False,
            'infer_types': True,
            'annotation_typing': False,
        },
        quiet=True,  # Suppress build output
        annotate=False,  # Disable generation of HTML annotation file
    ),
    zip_safe=False,
)
