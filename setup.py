import sys
from cx_Freeze import setup, Executable

# Minimal build options to avoid recursion issues
build_options = {
    'packages': [],  # Let cx_Freeze auto-detect
    'excludes': [
        'tkinter', 
        'matplotlib', 
        'scipy', 
        'test',
        'unittest',
        'distutils',
        'setuptools'
    ],
    'include_files': [
        ('model', 'model'),
        ('assets', 'assets'),
    ],
    'zip_include_packages': [],  # Don't zip anything
    'zip_exclude_packages': ['*'],  # Keep everything unzipped
    'optimize': 0,  # No optimization to avoid issues
}

# Use console base first to see any errors
base = None
if sys.platform == "win32":
    base = None  # Console app for debugging

executables = [
    Executable(
        'app.py',
        base=base,
        target_name='Pragati-AI.exe',
        icon='assets/icon.ico'
    )
]

setup(
    name='Pragati-AI',
    version='1.0',
    description='AI Application',
    options={'build_exe': build_options},
    executables=executables
)