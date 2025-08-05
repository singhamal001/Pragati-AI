# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[
        ('C:/Users/singh/anaconda3/envs/gemma3n/Lib/site-packages/llama_cpp/lib', 'llama_cpp/lib')
    ],
    datas=[
        ('./model', 'model'),
        ('./assets', 'assets'),
    ],
    hiddenimports=[
        'piper',
        'piper.voice',
        'llama_cpp',
        # NumPy modules
        'numpy.core._multiarray_umath',
        'numpy.core._multiarray_tests',
        'numpy.core._dtype_ctypes',
        'numpy.core._methods',
        'numpy.lib.format',
        'numpy.random.common',
        'numpy.random.bounded_integers',
        'numpy.random.entropy',
        'numpy.random._common',
        'numpy.random._bounded_integers',
        'numpy.random._entropy',
        'numpy.random._generator',
        'numpy.random._mt19937',
        'numpy.random._pcg64',
        'numpy.random._philox',
        'numpy.random._sfc64',
        'numpy.random.bit_generator',
        'numpy.random.mtrand',
        'numpy.core._internal',
        'numpy.core._dtype',
        'numpy.core._exceptions',
        'numpy.core._asarray',
        'numpy.core._ufunc_config',
    ],
    hookspath=['./'],  # Look for hooks in current directory
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Pragati-AI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Temporarily enable console for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Pragati-AI'
)