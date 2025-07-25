# --- Model and Path Configurations ---
WHISPER_MODEL_NAME = "openai/whisper-base.en"
PIPER_MODEL_PATH = './model/en_US-hfc_female-medium.onnx'
GEMMA_MODEL_NAME = 'gemma3n:e4b'

# --- Hardware and Performance ---
# Use "cuda" if you have a compatible NVIDIA GPU, otherwise "cpu"
DEVICE = "cpu"

# --- Interview Flow Control ---
MAX_QUESTIONS = 6  # The interview will have an intro + this many questions

# --- Audio Processing Settings ---
MIC_SAMPLE_RATE = 16000
PAUSE_THRESHOLD = 4.0 # Seconds of silence to end a user's turn