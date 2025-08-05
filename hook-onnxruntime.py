from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs
import os

# Collect everything from onnxruntime
datas, binaries, hiddenimports = collect_all('onnxruntime')

# Collect everything from numpy to avoid missing modules
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all('numpy')
datas += numpy_datas
binaries += numpy_binaries
hiddenimports += numpy_hiddenimports

# Add specific hidden imports that are commonly missed
hiddenimports += [
    'onnxruntime.capi._pybind_state',
    'onnxruntime.capi.onnxruntime_pybind11_state',
    'onnxruntime.backend.backend',
    'numpy.core._methods',
    'numpy.lib.format',
    'numpy.core._multiarray_umath'
]

# Collect all dynamic libraries
binaries += collect_dynamic_libs('onnxruntime')
binaries += collect_dynamic_libs('numpy')

# Collect all data files
datas += collect_data_files('onnxruntime')
datas += collect_data_files('numpy')