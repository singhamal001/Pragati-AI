import sys
from cx_Freeze import setup, Executable

# This list contains packages that cx_Freeze might not find automatically,
# but which don't need special hooks.
# Hooks will automatically handle llama_cpp and onnxruntime.
packages_to_include = [
    "re",
    "numpy",
    "pydantic",
    "pydub",
    "sounddevice",
    "speech_recognition",
    "whisper",
    "piper",
    "customtkinter"
]

# This is the most important part for your models and assets.
# It tells cx_Freeze to copy these entire directories into the final build.
# The format is ('source_path', 'destination_path_in_build')
files_to_include = [
    ('model', 'model'),
    ('assets', 'assets'),
]

# Build options
build_options = {
    "packages": packages_to_include,
    "excludes": ["tkinter", "test", "unittest", "pandas"], # Exclude what isn't needed
    "include_files": files_to_include,
    "include_msvcr": True, # Important for including Microsoft Visual C++ Redistributable on Windows
}

# --- Base Executable ---
# "Win32GUI" hides the black console window for a clean user experience.
# Set to None instead of "Win32GUI" if you want the console to show for debugging.
base = None
if sys.platform == "win32":
    base = "Win32GUI"

main_executable = Executable(
    "app.py",
    base=base,
    target_name="Pragati-AI.exe",
    icon="assets/icon.ico" # Make sure this icon file exists
)

# --- Run the Setup ---
setup(
    name="Pragati AI - Interview Coach",
    version="1.0",
    description="An AI-powered interview coach for visually impaired users.",
    options={"build_exe": build_options},
    executables=[main_executable]
)
